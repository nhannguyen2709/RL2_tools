"""
Override for Qwen tokenizers to work with incremental tokenization.
"""

def get_simple_qwen_template():
    """Returns modifed chat template for Qwen tokenizer."""
    return (
        "{%- if tools %}"
        "{{- '<|im_start|>system\\n' }}"
        "{%- if messages[0].role == 'system' %}"
        "{{- messages[0].content + '\\n\\n' }}"
        "{%- endif %}"
        "{{- '# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}"
        "{%- for tool in tools %}"
        "{{- '\\n' }}"
        "{{- tool | tojson }}"
        "{%- endfor %}"
        "{{- '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}"
        "{%- else %}"
        "{%- if messages[0].role == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- endif %}"
        
        # Simple message processing without complex logic
        "{%- for message in messages %}"
        "{%- if message.content is string %}"
        "{%- set content = message.content %}"
        "{%- else %}"
        "{%- set content = '' %}"
        "{%- endif %}"
        
        # User messages
        "{%- if message.role == 'user' %}"
        "{{- '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}"
        
        # Assistant messages - preserve content as-is
        "{%- elif message.role == 'assistant' %}"
        "{{- '<|im_start|>assistant\\n' + content }}"
        "{%- if message.tool_calls %}"
        "{%- for tool_call in message.tool_calls %}"
        "{{- '\\n<tool_call>\\n{\"name\": \"' }}"
        "{{- tool_call.function.name }}"
        "{{- '\", \"arguments\": ' }}"
        "{%- if tool_call.function.arguments is string %}"
        "{{- tool_call.function.arguments }}"
        "{%- else %}"
        "{{- tool_call.function.arguments | tojson }}"
        "{%- endif %}"
        "{{- '}\\n</tool_call>' }}"
        "{%- endfor %}"
        "{%- endif %}"
        "{{- '<|im_end|>\\n' }}"
        
        # Tool messages - simple format
        "{%- elif message.role == 'tool' %}"
        "{{- '<|im_start|>user\\n<tool_response>\\n' + content + '\\n</tool_response><|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
        
        # Generation prompt
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    )
