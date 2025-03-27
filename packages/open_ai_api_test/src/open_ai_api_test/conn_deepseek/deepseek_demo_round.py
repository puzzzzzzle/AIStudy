from openai import OpenAI

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from key_loadder import load_key

api_key, base_url = load_key()

# 初始化OpenAI客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)
# 通过 messages 数组实现上下文管理
messages = [
    {'role': 'user', 'content': '你好'},
    {'role': 'assistant', 'content': '你好，有什么可以帮助你的？'},
    {'role': 'user', 'content': '9.9和9.11哪个大'}
]

completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=messages
)

print("=" * 20 + "第一轮对话" + "=" * 20)
# 通过reasoning_content字段打印思考过程
print("=" * 20 + "思考过程" + "=" * 20)
print(completion.choices[0].message.reasoning_content)
# 通过content字段打印最终答案
print("=" * 20 + "最终答案" + "=" * 20)
print(completion.choices[0].message.content)

messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
messages.append({'role': 'user', 'content': '你是谁'})
print("=" * 20 + "第二轮对话" + "=" * 20)
completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=messages
)
# 通过reasoning_content字段打印思考过程
print("=" * 20 + "思考过程" + "=" * 20)
print(completion.choices[0].message.reasoning_content)
# 通过content字段打印最终答案
print("=" * 20 + "最终答案" + "=" * 20)
print(completion.choices[0].message.content)
