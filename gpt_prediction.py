import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score, classification_report
from astropy.table import Table
from sklearn.metrics import roc_auc_score
import os
import json
from langchain_openai import ChatOpenAI
from utils import predata
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI()

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

df = pd.read_csv('student-data.csv')
data = df.to_numpy()
n = data.shape[1]
x = data[:,0:n-1]
y = data[:,n-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=41)
# print("x_train,x_test,y_train,y_test",type(x_train),type(x_test),type(y_train),type(y_test))
# print(x_train.shape)
print(x_train[0,:])
for i in x_train.shape[0]:
    messages = [
        SystemMessage(content="1.你是一个军事数据分析师。请从以下描述中提取并总结'作战目标'，'作战数量'，'作战对象'，'损失要求'，'作战区域'和'完成期限'，'气候条件'，'温度'，'昼夜区分'，'目标状态'，'海况情况'\
                               2.修改后的description的文字中必须明确包含上述''中的字段，若没有则在字段后表示无或者合理的捏造数据\
                               3.下面为示例：作战目标为，作战对象为，损失要求为，作战区域为，完成期限为，气候条件为，温度为，昼夜区分为，目标状态为，海况情况为。\
                               4.在所给描述中提取出3.中的字段，若无信息提取则表示无或者合理的捏造数据"),
        HumanMessage(content=f"描述: {description}")
    ]
    response = llm.invoke(messages)
    


with open('output.json', 'w', encoding='utf-8') as file:
    json.dump(processed_blocks, file, ensure_ascii=False, indent=4)

print("JSON 文件处理完成，并已保存到 output.json")
