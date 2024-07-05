
from typing import Dict
import os
import yaml
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI

# 获取当前相对文件路径
current_dir = os.path.dirname(os.path.abspath(__file__)) #()内部文件的父路径
config_path = os.path.join(current_dir, 'config.yml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f) #解析为python对象

# 加载 .env 到环境变量
os.environ["OPENAI_API_KEY"] = config['api_key']


# 聊天接口
def chat(response_schemas, systemPrompt, humanPrompt):
    chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5)

    # 构建返回解析器
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions(only_json=True)

    # 构建prompt
    systemTemplate = systemPrompt + "\n请尽可能好的回答提问。\n{format_instructions}"
    humanTemplate = humanPrompt.replace("{", "[").replace("}", "]")
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(systemTemplate),
            # 防止humanPrompt中使用了花括号，被误解为模板变量
            HumanMessagePromptTemplate.from_template(humanTemplate)
        ],
        # input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    # 格式化输出
    _input = prompt.format_prompt()
    output = chat_model.invoke(_input.to_messages())
    # print(output.content)
    # print(output.content.replace('json', ''))
    result = output.content.replace('json', '')
    return output_parser.parse(result)
