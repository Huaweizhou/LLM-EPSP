#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author zhouhuawei time:2024/6/24

import os
import json
from langchain_openai import ChatOpenAI
from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = "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"
# 30$私钥
# os.environ["OPENAI_API_KEY"] = "sk-2EDY9F7TMNULFOhV890953EcBd93437dA534865745366e57"
os.environ["OPENAI_API_KEY"] = "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"


llm = ChatOpenAI(model="gpt-4")

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
messages = [
    SystemMessage(content="1.你是一个军事数据分析师。请从以下描述中提取并总结'作战目标'，'作战数量'，'作战对象'，'损失要求'，'作战区域'和'完成期限'，'气候条件'，'温度'，'昼夜区分'，'目标状态'，'海况情况'\
                            2.修改后的description的文字中必须明确包含上述''中的字段，若没有则在字段后表示无或者合理的捏造数据\
                            3.下面为示例：作战目标为，作战对象为，损失要求为，作战区域为，完成期限为，气候条件为，温度为，昼夜区分为，目标状态为，海况情况为。\
                            4.在所给描述中提取出3.中的字段，若无信息提取则表示无或者合理的捏造数据"),
    HumanMessage(content=f"随便说点")
]
response = llm.invoke(messages)
print("response")