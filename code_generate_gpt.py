#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author zhouhuawei time:2024/4/25
import pandas as pd
from gpt import chat
from code_utils import *
import random


response_schemas = [
    ResponseSchema(name="question", description="所出题目题干"),
    ResponseSchema(name="answer", description="所出题目答案"),
    ResponseSchema(name="knowledge", description="所出题目核心知识点"),
    ResponseSchema(name="testcase", description="所出题目测试示例演示(请给出5个演示示例)"),
    ResponseSchema(name="analyze", description="所出题目解析"),
    ResponseSchema(name="grade", description="所出题目的难度"),
    ResponseSchema(name="input_module", description="所出题目的答案的输入模板")
]


data = pd.read_excel('D:/知识体系_知识图谱.xlsx')
df = pd.DataFrame()

for index, row in data.iterrows():
    value1 = row['知识点名称']
    value2 = row['知识点描述']
    systemPrompt ="你是一位编程题的出题专家，现在需要你按照下面的要求出关于python知识图谱的编程题，答案请尽可能准确且为json格式。"
    humanPrompt = f"首先对{value1},{value2}的内容进行学习，确定其所考察的python知识图谱的知识点"\
                  f"请严格根据上面的知识点，即{value1},{value2}帮我出一个高质量的编程题"\
                  f"出题难度标准请参照{difficulty_standard(random.randint(0, 2))}" \
                  f"出题规范标准请参照{basic_requirement}" \
                  f"相似度标准请参考{getSimilarRules()}"\
                  f"试题的回答模板请严格参考{format_instructinos}"

    response_dict = chat(response_schemas, systemPrompt, humanPrompt)
    if len(df) == 0:
        for key, value in response_dict.items():
            df[key] = [value]
    else:
        new_row = {}
        new_row_list = [new_row]
        for key, value in response_dict.items():
            new_row[key] = value
        # print("new_row_list", new_row)
        # print("pd.DataFrame(new_row)", pd.DataFrame(new_row_list))
        df = pd.concat([df, pd.DataFrame(new_row_list)], ignore_index=True)

    # 保存表格为Excel文件
    excel_file = 'D:/code_output1.xlsx'  # Excel文件名和路径
    df.to_excel(excel_file, index=False)
