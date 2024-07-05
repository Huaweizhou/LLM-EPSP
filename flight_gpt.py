#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author zhouhuawei time:2024/6/24

import os
import json
from langchain_openai import ChatOpenAI

# os.environ["OPENAI_API_KEY"] = "sk-FdRhcsnAOevePNvruOQsEBL8UFZqEUZr65yN4bXodyJlYaeM"
# os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI()

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# 读取 JSON 文件
with open('plan-1.json', 'r', encoding='utf-8') as file:
    json_blocks = json.load(file)

for block in json_blocks:
    description = block["document"]["plan"]["description"]
    messages = [
        SystemMessage(content="1.你是一个军事数据分析师。请从以下描述中提取并总结'作战目标'，'作战数量'，'作战对象'，'损失要求'，'作战区域'和'完成期限'，'气候条件'，'温度'，'昼夜区分'，'目标状态'，'海况情况'\
                               2.修改后的description的文字中必须明确包含上述''中的字段，若没有则在字段后表示无或者合理的捏造数据\
                               3.下面为示例：作战目标为，作战对象为，损失要求为，作战区域为，完成期限为，气候条件为，温度为，昼夜区分为，目标状态为，海况情况为。\
                               4.在所给描述中提取出3.中的字段，若无信息提取则表示无或者合理的捏造数据"),
        HumanMessage(content=f"描述: {description}")
    ]
    response = llm.invoke(messages)
    block["document"]["plan"]["description"] = response.content.strip(' ').strip('\n')
    # print(response)
    # response_dict = json.loads(response.content.strip(' ').strip('```json').strip('```').strip('\n').strip(''))
    # print(response_dict)
processed_blocks = json_blocks

with open('output.json', 'w', encoding='utf-8') as file:
    json.dump(processed_blocks, file, ensure_ascii=False, indent=4)

print("JSON 文件处理完成，并已保存到 output.json")