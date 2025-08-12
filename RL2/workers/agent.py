import json
import logging
import torch
import asyncio
import importlib.util
import sys
from collections import defaultdict
from typing import List, Dict, Optional, Literal, Any
from json import JSONDecodeError
from pathlib import Path
from enum import Enum
from copy import deepcopy

from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import PreTrainedTokenizer
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.entrypoints.openai.protocol import Tool

from RL2.workers.rollout import Rollout
from RL2.workers.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)
from RL2.workers.fix_qwen_chat_template import get_simple_qwen_template

logger = logging.getLogger(__name__)


DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the question. First, reason privately inside <think>...</think>. "
    "If you need external knowledge: \n"
    "- Use web_search to find candidate results with short previews: \n"
    "  <tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"...\"}}</tool_call> \n"
    "  The system will return top results between <tool_response>...</tool_response>.\n"
    "- After deciding which URL to open, fetch the full content with web_visit: \n"
    "  <tool_call>{\"name\": \"web_visit\", \"arguments\": {\"url\": \"https://...\"}}</tool_call>\n"
    "You may repeat web_search and web_visit as needed. When you are confident, output only: \n"
    "<answer> your final answer here </answer>.\n"
)


def get_tool_call_parser_type(tokenizer) -> str:
    """Get the appropriate tool call parser type for the tokenizer."""
    return "qwen25"


def compute_position_id_with_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute position IDs from attention mask."""
    return torch.cumsum(attention_mask, dim=-1) - 1


class RequestStateEnum(Enum):
    """Request state enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    TOOL_CALLING = "tool_calling"
    COMPLETED = "completed"


class FinishReasonTypeEnum(Enum):
    """Finish reason type enumeration."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    
    @classmethod
    def from_str(cls, value: str):
        """Convert string to enum."""
        mapping = {
            "stop": cls.STOP,
            "length": cls.LENGTH,
            "tool_call": cls.TOOL_CALL,
        }
        return mapping.get(value, cls.STOP)


class Message(BaseModel):
    """Message data model."""
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AgentRolloutRequest(BaseModel):
    """Agent rollout request data model similar to AsyncRolloutRequest."""
    
    request_id: str
    state: RequestStateEnum
    messages: List[Message]
    tools: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    reward_scores: Dict[str, float] = {}
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}
    answer: str = ""
    user_input: str = ""
    
    format_config: dict = {
        "qwen": {
            "assistant_prefix_msg": "<|im_start|>assistant\n",
            "assistant_suffix_msg": "<|im_end|>\n",
            "merge_tool_response": True,
            "tool_prefix_msg": "<|im_start|>user\n",
            "tool_suffix_msg": "<|im_end|>\n",
            "tool_response_prefix_msg": "<tool_response>\n",
            "tool_response_suffix_msg": "\n</tool_response>",
        },
    }
    
    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        """Get generation prompt token IDs."""
        return tokenizer.apply_chat_template(
            conversation=[msg.model_dump() for msg in self.messages],
            tools=[tool.model_dump() for tool in self.tools] if self.tools else None,
            add_generation_prompt=True,
            tokenize=True,
        )
    
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
        format: Literal["chatml", "qwen"] = "qwen",
        already_over_long: bool = False,
    ) -> None:
        """Add assistant message and update token sequences."""
        msg = Message(role="assistant", content=content, tool_calls=tool_calls)
        self.messages.append(msg)        
        if tool_calls is not None:
            content_with_tool_calls: str = tokenizer.apply_chat_template(
                [msg.model_dump()], add_generation_prompt=False, tokenize=False
            )
        else:
            content_with_tool_calls = content

        if format in self.format_config:
            prefix_msg = self.format_config[format]["assistant_prefix_msg"]
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            suffix_msg = self.format_config[format]["assistant_suffix_msg"]
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            if tool_calls is not None:
                content = content_with_tool_calls.split(f"{prefix_msg}")[-1].split(f"{suffix_msg}")[0]
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            if self.input_ids[-len(prefix_token_ids):] == prefix_token_ids:
                append_token_ids = content_token_ids
                _loss_mask = [1] * len(content_token_ids)
            elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
                append_token_ids = prefix_token_ids + content_token_ids
                _loss_mask = [0] * len(prefix_token_ids) + [1] * len(content_token_ids)
            else:
                max_len = max(len(prefix_token_ids), len(suffix_token_ids))
                raise ValueError(
                    f"Unsupported end of message format: "
                    f"{tokenizer.decode(self.input_ids[-max_len:])}, "
                    f"{tokenizer.decode(self.input_ids)=}, {self.messages=}"
                )
                
            if not already_over_long:
                append_token_ids += suffix_token_ids
                _loss_mask += [1] * len(suffix_token_ids)

            # print(tokenizer.decode(self.input_ids + append_token_ids))
            # import pdb; pdb.set_trace()

            self.input_ids += append_token_ids
            _attention_mask = [1] * len(append_token_ids)
            self.attention_mask += _attention_mask
            _delta_position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
            last_position_id = self.position_ids[-1]
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            self.loss_mask += _loss_mask
            self.position_ids += _position_ids
        else:
            raise ValueError(f"Unsupported format: {format}")

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), \
            f"Request {self.request_id} has different lengths: " \
            f"{len(self.input_ids)=}, {len(self.attention_mask)=}, " \
            f"{len(self.position_ids)=}, {len(self.loss_mask)=}"
    
    def add_tool_response_message(
        self, 
        tokenizer: PreTrainedTokenizer, 
        content: str, 
        last_tool: bool, 
        format: Literal["chatml", "qwen"] = "qwen"
    ) -> None:
        """Add tool response message and update token sequences."""
        msg = Message(role="tool", content=content)
        self.messages.append(msg)
        
        if format in self.format_config:
            merge_tool_responses = self.format_config[format].get("merge_tool_response", False)
            prefix_msg = self.format_config[format]["tool_prefix_msg"]
            prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
            suffix_msg = self.format_config[format]["tool_suffix_msg"]
            suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
            prefix_resp = self.format_config[format].get("tool_response_prefix_msg", "")
            prefix_resp_token_ids = tokenizer.encode(prefix_resp, add_special_tokens=False)
            suffix_resp = self.format_config[format].get("tool_response_suffix_msg", "")
            suffix_resp_token_ids = tokenizer.encode(suffix_resp, add_special_tokens=False)
            
            full_suffix_token_ids = suffix_resp_token_ids + (suffix_token_ids if last_tool or not merge_tool_responses else [])
            content_token_ids = tokenizer.encode(content, add_special_tokens=False)
            
            if (self.input_ids[-len(prefix_token_ids):] == prefix_token_ids or 
                self.input_ids[-len(suffix_resp_token_ids):] == suffix_resp_token_ids):
                append_token_ids = prefix_resp_token_ids + content_token_ids + full_suffix_token_ids
            elif self.input_ids[-len(prefix_resp_token_ids):] == prefix_resp_token_ids:
                append_token_ids = content_token_ids + full_suffix_token_ids
            elif self.input_ids[-len(suffix_token_ids):] == suffix_token_ids:
                append_token_ids = prefix_token_ids + prefix_resp_token_ids + content_token_ids + full_suffix_token_ids
            else:
                raise ValueError(f"Unsupported end of message format: {tokenizer.decode(self.input_ids[-len(prefix_token_ids):])}")
                
            self.input_ids += append_token_ids
            _attention_mask = [1] * len(append_token_ids)
            self.attention_mask += _attention_mask
            _delta_position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
            last_position_id = self.position_ids[-1]
            _position_ids = [pos_id + last_position_id for pos_id in _delta_position_ids]
            self.loss_mask += [0] * len(append_token_ids)
            self.position_ids += _position_ids
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), \
            f"Request {self.request_id} has different lengths: " \
            f"{len(self.input_ids)=}, {len(self.attention_mask)=}, " \
            f"{len(self.position_ids)=}, {len(self.loss_mask)=}"
    
    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """Update metrics for a specific tool."""
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)
    
    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward: float,
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        """Finalize the request and calculate final sequences."""
        self.state = RequestStateEnum.COMPLETED
        self.reward_scores["final"] = reward
        self.response_ids = self.input_ids[len(self.prompt_ids):]
        
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
            
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), \
            f"Request {self.request_id} has different lengths: " \
            f"{len(self.input_ids)=}, {len(self.attention_mask)=}, " \
            f"{len(self.position_ids)=}, {len(self.loss_mask)=}"
    
    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        """Truncate sequences to maximum lengths."""
        self.input_ids = self.input_ids[:self.max_model_len]
        self.attention_mask = self.attention_mask[:self.max_model_len]
        self.position_ids = self.position_ids[:self.max_model_len]
        self.loss_mask = self.loss_mask[:self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids):][:self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask):][:self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids):][:self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask):][:self.max_response_len]
    
    def to_training_data(self) -> Dict[str, torch.Tensor]:
        """Convert to training data format."""
        return {
            "states": torch.LongTensor(self.input_ids[:-1]),
            "actions": torch.LongTensor(self.input_ids[1:]),
            "action_mask": torch.LongTensor(self.loss_mask[1:]),
            "position_ids": torch.LongTensor(self.position_ids[:-1]),
            "rewards": torch.FloatTensor((len(self.input_ids) - 2) * [0] + [self.reward_scores.get("final", 0.0)])
        }


class Agent(Rollout):
    """
    Tools-enabled agent that inherits from Rollout.
    
    Extends the base Rollout functionality with tool-calling capability,
    using SGLang's built-in function call parsing for better integration.
    """

    def __init__(self, config):
        super().__init__(config)
        
        self.tokenizer.chat_template = get_simple_qwen_template()

        # Initialize tools from configuration
        (
            self.tool_schemas,
            self.tool_map,
            self.tool_call_parser_type,
            self.tools,
            self.function_parser,
        ) = self._initialize_tools(config, self.tokenizer)
        
        # Log tool initialization
        logger.info(f"Loaded {len(self.tool_map)} tools: {list(self.tool_map.keys())}")
        for tool_name, tool in self.tool_map.items():
            if hasattr(tool, 'search_url'):
                logger.info(f"Tool '{tool_name}' initialized with search URL: {tool.search_url}")
            else:
                logger.info(f"Tool '{tool_name}' initialized")

    def _initialize_tools(self, config, tokenizer):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.tools_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.managers.io_struct.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        tools_config_path = getattr(config, 'tools_config_path', None)
        if not tools_config_path:
            raise ValueError("tools_config_path must be specified in config")

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []

            for tool_name, tool_config in tools_config.tools.tool_instances.items():
                class_path = tool_config.class_path
                module_name, class_name = class_path.rsplit(".", 1)

                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    if spec is None:
                        raise ImportError(f"Cannot find module: {module_name}")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]

                tool_cls = getattr(module, class_name)

                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

                params_dict = OmegaConf.to_container(tool_config.params, resolve=True)
                tool = tool_cls(
                    config=params_dict,
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)

            return tool_list

        tools_config_file = Path(tools_config_path)
        if not tools_config_file.exists():
            raise FileNotFoundError(f"Tool configuration file not found: {tools_config_file}")
        
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    def _create_rollout_request(self, ex: Dict[str, Any]) -> AgentRolloutRequest:
        """Create and initialize a rollout request from input data."""
        messages, answer = ex["messages"], ex["answer"]
        user_input = messages[-1]["content"]
        
        # Add prefix to user content
        modified_messages = deepcopy(messages)
        modified_messages[-1]["content"] = DEFAULT_USER_CONTENT_PREFIX + modified_messages[-1]["content"]
        
        # Convert to Message objects
        message_objects = [Message(**msg) for msg in modified_messages]
        
        # Get initial prompt token IDs
        if self.config.apply_chat_template and self.function_parser:
            prompt_ids = self.tokenizer.apply_chat_template(
                conversation=[msg.model_dump() for msg in message_objects],
                tools=[tool.model_dump() for tool in self.tools],
                add_generation_prompt=False,
                tokenize=True,
            )
        elif self.config.apply_chat_template:
            prompt_ids = self.tokenizer.apply_chat_template(
                conversation=[msg.model_dump() for msg in message_objects],
                add_generation_prompt=False,
                tokenize=True,
            )
        else:
            prompt_text = "".join([msg.content for msg in message_objects])
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Initialize attention mask and position IDs
        attention_mask = [1] * len(prompt_ids)
        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
        loss_mask = [0] * len(prompt_ids)  # No loss on prompt tokens
        
        # Create request
        request = AgentRolloutRequest(
            request_id=f"rollout_{id(ex)}",
            state=RequestStateEnum.PENDING,
            messages=message_objects,
            tools=self.tool_schemas,
            input_ids=prompt_ids,
            prompt_ids=prompt_ids.copy(),
            response_ids=[],
            attention_mask=attention_mask,
            prompt_attention_mask=attention_mask.copy(),
            response_attention_mask=[],
            position_ids=position_ids,
            prompt_position_ids=position_ids.copy(),
            response_position_ids=[],
            loss_mask=loss_mask,
            prompt_loss_mask=loss_mask.copy(),
            response_loss_mask=[],
            answer=answer,
            user_input=user_input,
            max_response_len=getattr(self.config, 'max_response_len', 8192),
            max_model_len=getattr(self.config, 'max_model_len', 32768),
        )
        
        return request

    async def _handle_pending_state(self, request: AgentRolloutRequest) -> None:
        """Handle pending state - prepare for generation and initialize tools if needed."""
        # Initialize tools if they exist (similar to SGLang implementation)
        if request.tools is not None and len(request.tools) > 0:
            tool_creation_coroutines = []
            for tool_schema in request.tools:
                # Handle both dict and object schemas
                tool_name = tool_schema['function']['name'] if isinstance(tool_schema, dict) else tool_schema.function.name
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    # Only initialize tools that have a create method
                    if hasattr(tool, 'create') and callable(getattr(tool, 'create')):
                        create_kwargs = request.tools_kwargs.get(tool_name, {}).get("create_kwargs", {})
                        tool_creation_coroutines.append(tool.create(request.request_id, **create_kwargs))
            
            if tool_creation_coroutines:
                await asyncio.gather(*tool_creation_coroutines)

    async def _handle_engine_call(
        self, 
        request: AgentRolloutRequest,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle LLM engine call for generation."""
        generation_prompt_ids = request.get_generation_prompt(self.tokenizer)
        # Generate response
        response = await self.llm.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=self.train_sampling_params,
        )
        return response

    async def rollout(self, ex, train):
        """
        Rollout method using AgentRolloutRequest for proper tokenization handling.
        
        This follows the SGLangRollout pattern for better tool integration and 
        tokenizer template handling.
        """
        # Create rollout request
        request = self._create_rollout_request(ex)
        metric = defaultdict(list)
        
        current_turns = 0
        max_turns = getattr(self.config, 'max_turns', 10)
        finish_reason_type = None
        
        while current_turns < max_turns:
            if request.state == RequestStateEnum.PENDING:
                await self._handle_pending_state(request)
                request.state = RequestStateEnum.RUNNING
            elif request.state == RequestStateEnum.TOOL_CALLING:
                if request.messages[-1].tool_calls is not None:
                    parsed_tool_calls = request.messages[-1].tool_calls
                    
                    # Use new VERL interface: execute(instance_id, parameters, **kwargs) -> (response, reward, metrics)
                    tool_call_results = await asyncio.gather(
                        *[
                            self.tool_map[tool_call.function.name].execute(
                                request.request_id,  # instance_id
                                tool_call.function.arguments,  # parameters as dict
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    
                    for i, (tool_call, (response, reward, metrics)) in enumerate(zip(parsed_tool_calls, tool_call_results)):
                        # Parse tool result from the new tuple format
                        if isinstance(response, str):
                            try:
                                parsed_result = json.loads(response)
                                tool_name = parsed_result.get("tool_name", tool_call.function.name)
                                
                                if parsed_result.get("success", False):
                                    content = parsed_result.get("data", "")
                                else:
                                    error_msg = parsed_result.get("error", "Unknown error")
                                    content = f"[{tool_name} Error]: {error_msg}"
                            except json.JSONDecodeError:
                                # If response is not JSON, use it directly
                                content = response
                        else:
                            content = str(response)
                        
                        # Add tool response message with proper tokenization
                        request.add_tool_response_message(
                            self.tokenizer,
                            content,
                            (i == len(parsed_tool_calls) - 1),
                            format="qwen",  # Use qwen format for tool responses
                        )
                        
                        # Update metrics from the tool execution
                        if metrics:
                            request.update_metrics(metrics, tool_call.function.name)
                            
                        # Check if we're approaching max length
                        if len(request.input_ids) >= request.max_model_len:
                            break
                            
                    if len(request.input_ids) >= request.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                        
                    request.state = RequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {request.messages[-1]}")
                    
            elif request.state == RequestStateEnum.RUNNING:
                output = await self._handle_engine_call(request)
                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                
                # Track metrics
                meta_info = output["meta_info"]
                metric["response_length"].append(meta_info["completion_tokens"])
                metric["length_clip_ratio"].append(
                    meta_info["finish_reason"]["type"] == "length"
                )
                
                # Extract content limited by completion tokens
                content = self.tokenizer.decode(
                    self.tokenizer.encode(
                        content, add_special_tokens=False
                    )[:meta_info["completion_tokens"]]
                )
                
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    request.add_assistant_message(
                        self.tokenizer,
                        content,
                        already_over_long=True,
                        format="qwen",
                    )
                    break
                else:
                    # Check for tool calls
                    if self.function_parser and self.function_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        request.state = RequestStateEnum.TOOL_CALLING
                        
                        try:
                            normed_content, tool_calls = self.function_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                            
                        # Process and validate tool calls
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                            
                        if len(parsed_tool_calls) > 0:
                            request.add_assistant_message(
                                self.tokenizer,
                                normed_content,
                                tool_calls=parsed_tool_calls,
                                format="qwen",
                            )
                        else:
                            request.add_assistant_message(
                                self.tokenizer,
                                content,
                                format="qwen",
                            )
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            request.state = RequestStateEnum.COMPLETED
                            break
                    else:
                        request.add_assistant_message(
                            self.tokenizer,
                            content,
                            format="qwen",
                        )
                        break

        if current_turns >= max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate reward
        reward = self.env.llm_reward_fn(
            [msg.model_dump() for msg in request.messages], 
            request.answer, 
            request.user_input
        )

        # Finalize request
        request.finalize(self.tokenizer, reward, finish_reason_type)

        # Get training data in the expected format
        training_data = request.to_training_data()

        # Track metrics
        metric["n_turns"].append(current_turns)
        metric["rewards"].append(reward)
        metric["trajectory_length"].append(len(training_data["states"]))

        return training_data, [msg.model_dump() for msg in request.messages], metric


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from tqdm.asyncio import tqdm
    import torch.distributed as dist

    from RL2.utils.comm import initialize_global_process_group
    from RL2.utils.comm import split_and_scatter_list

    initialize_global_process_group()

    config = OmegaConf.load("RL2/trainer/config/ppo_searchr1.yaml")
    rollout = Agent(config.rollout)
    data_list = [
        {
            "messages": [
                {"role": "user", "content": "How many episodes are there in Big Little Lies Season 2?"}
            ],
            "answer": "Seven"
        },
        {
            "messages": [
                {"role": "user", "content": "What is the date of birth of the performer of song Toinen?"}
            ],
            "answer": "20 March 1983"
        },
        {
            "messages": [
                {"role": "user", "content": "Who was sacked as Cardiff City manager on December 27th, 2013?"}
            ],
            "answer": "Malky Mackay was the manager sacked on that day."
        },
        {
            "messages": [
                {"role": "user", "content": "What is the place of birth of the performer of song Russian Roulette (Rihanna Song)?"}
            ],
            "answer": "Saint Michael, Barbados"
        },
        # {
        #     "messages": [
        #         {"role": "user", "content": "After a stage of the 2005 Tour de France, which had Robbie McEwen as the sprint finish winner and included a breakaway rider who rode solo for 160km, in the general classification:\n\nThe rider who holds the record for the most Tour de France participations, is of the same nationality as the winner of the women's individual time trial at the World Road Racing Championship (which was the first to include the women's individual time trial), and rode for the team that was later rebranded to the team of the 2008 Tour de France winner, is behind another rider by how many seconds?\n\nThis other rider also holds the record for most Tour de France participations, is of the same nationality, won the men's road race at the World Road Racing Championship (which was the first to include the women's team time trial), won the overall classification of the Tour DuPont that was the first to include a route through South Carolina, and was the first American to win that World Championship.\n\nGiven that the points classification was led by a German rider, by how many seconds is the first rider behind the second rider?"}
        #     ],
        #     "answer": "55 seconds"
        # }
    ]
    if rollout.device_mesh["tp"].get_local_rank() == 0:
        data_list = split_and_scatter_list(
            data_list, rollout.device_mesh["dp"]
        )
        loop = asyncio.get_event_loop()
        outputs = loop.run_until_complete(
            tqdm.gather(
                *(rollout.rollout(ex, True) for ex in data_list),
                desc="Rollout", position=1, leave=False,
                disable=(dist.get_rank() != 0)
            )
        )
        data, all_messages, metrics = map(list, zip(*outputs))
        metrics = {
            f"{k}/train": sum([metric[k] for metric in metrics], [])
            for k in metrics[0].keys()
        }
        import pdb; pdb.set_trace()
        print(f"Device mesh rank: {rollout.device_mesh['dp'].get_local_rank()}")
        print(f"Metrics: {metrics}")
        print("Rollout completed successfully!")
        print(outputs)
        print("=" * 80)
    
    dist.destroy_process_group()