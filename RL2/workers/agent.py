import json
import logging
import torch
import asyncio
import importlib.util
import sys
from collections import defaultdict
from typing import List, Dict, Optional
from json import JSONDecodeError
from pathlib import Path

from omegaconf import OmegaConf
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.entrypoints.openai.protocol import Tool

from RL2.datasets import tokenize_messages
from RL2.workers.rollout import Rollout
from RL2.workers.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
)

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


from pydantic import BaseModel
class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class Agent(Rollout):
    """
    Tools-enabled agent that inherits from Rollout.
    
    Extends the base Rollout functionality with tool-calling capability,
    using SGLang's built-in function call parsing for better integration.
    """

    def __init__(self, config):
        super().__init__(config)
        
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

                params_dict = OmegaConf.to_container(tool_config.params, resolve=True) if hasattr(tool_config, 'params') else {}
                tool = tool_cls(
                    config=params_dict,
                    tool_schema=tool_schema.get_openai_tool_schema(),
                )
                tool_list.append(tool)

            return tool_list

        tools_config_file = Path(tools_config_path)
        if not tools_config_file.exists():
            raise FileNotFoundError(f"Tool configuration file not found: {tools_config_file}")
        
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        print(f"Using parser type: {tool_call_parser_type}")
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

    async def rollout(self, ex, train):
        """
        Override rollout method to include search tool handling with SGLang function call parsing.
        
        This follows the SGLangRollout pattern for proper tool integration.
        """
        messages, answer = ex["messages"], ex["answer"]
        user_input = messages[-1]["content"]
        metric = defaultdict(list)
        
        current_turns = 0
        max_turns = getattr(self.config, 'max_turns', 10)

        messages[-1]["content"] = DEFAULT_USER_CONTENT_PREFIX + messages[-1]["content"]
        
        while current_turns < max_turns:
            # Prepare prompt with tools if function parser is available
            if self.config.apply_chat_template and self.function_parser:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tools=[tool.model_dump() for tool in self.tools],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            elif self.config.apply_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt = "".join([
                    msg["content"] for msg in messages
                ])
            
            # Generate response using async generation
            response = await self.llm.async_generate(
                prompt,
                sampling_params=self.train_sampling_params
                if train else self.test_sampling_params
            )

            meta_info = response["meta_info"]
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )

            # Extract content from response
            content = self.tokenizer.decode(
                self.tokenizer.encode(
                    response["text"], add_special_tokens=False
                )[:meta_info["completion_tokens"]]
            )
            
            # Determine finish reason and handle tool calls
            finish_reason_type = meta_info["finish_reason"]["type"]
            current_turns += 1
            
            if finish_reason_type == "length":
                # Add assistant message and break due to length limit
                messages.append({"role": "assistant", "content": content})
                break
            else:
                # Other finish reasons (interrupted, partial, etc.) - check for tool calls
                if self.function_parser and self.function_parser.has_tool_call(content):
                    # Parse tool calls from content
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
                    # Execute valid parsed tool calls
                    if len(parsed_tool_calls) > 0:
                        messages.append({"role": "assistant", "content": content})
                        tool_call_results = await asyncio.gather(
                            *[
                                self.tool_map[tool_call.function.name].execute(
                                    tool_call.function.arguments,
                                )
                                for tool_call in parsed_tool_calls
                            ]
                        )
                        # Format tool responses with standardized handling
                        tool_response_prefix_msg = "<tool_response>\n"
                        tool_response_suffix_msg = "\n</tool_response>"
                        tool_calls_content = ""
                        
                        for i, (tool_call, result) in enumerate(zip(parsed_tool_calls, tool_call_results)):
                            parsed_result = json.loads(result)
                            tool_name = parsed_result.get("tool_name", tool_call.function.name)
                            
                            if parsed_result.get("success", False):
                                # Successful tool execution
                                tool_calls_content += parsed_result.get("data", "")
                            else:
                                # Tool execution failed
                                error_msg = parsed_result.get("error", "Unknown error")
                                tool_calls_content += f"[{tool_name} Error]: {error_msg}"
                                
                            # Add spacing between multiple tool results
                            if i < len(parsed_tool_calls) - 1:
                                tool_calls_content += "\n\n"
                                    
                        tool_calls_content = tool_response_prefix_msg + tool_calls_content + tool_response_suffix_msg
                        messages.append({"role": "user", "content": tool_calls_content})
                    else:
                        messages.append({"role": "assistant", "content": content})
                        break
                else:
                    messages.append({"role": "assistant", "content": content})
                    break

        # Check for max turns limit
        if current_turns >= max_turns:
            logger.info(f"Reached maximum turns limit: {max_turns}")

        # Calculate reward
        reward = self.env.llm_reward_fn(messages, answer, user_input)

        # Tokenize messages for training
        ex = tokenize_messages(
            self.tokenizer,
            messages,
            self.config.apply_chat_template
        )
        ex.update({
            "rewards": torch.FloatTensor((ex["states"].shape[-1] - 1) * [0] + [reward])
        })

        metric["n_turns"].append(current_turns)
        metric["rewards"].append(reward)
        metric["trajectory_length"].append(len(ex["states"]))

        return ex, messages, metric


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
        }
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
        print(rollout.device_mesh["dp"].get_local_rank())
        print(outputs)
        print("=" * 80)
    
    dist.destroy_process_group()