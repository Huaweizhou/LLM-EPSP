import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from zhipuai import ZhipuAI
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, mean_squared_error

DATA_PATH = "dataset\dataset_math\student-data.csv"
DATA_PATH_TEST = "dataset\dataset_math\student-data-test.csv"

IN_COLS = [
    "school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob",\
    "Fjob","reason","guardian","traveltime","studytime","failures","schoolsup",\
    "famsup","paid","activities","nursery","higher","internet","romantic","famrel",\
    "freetime","goout","Dalc","Walc","health","absences","passedsex","age","address",\
    "famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime",\
    "studytime","failures","schoolsup","famsup","paid","activities","nursery","higher",\
    "internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences","passed"
]

#########################################################################################################
'''
api相关的属性
'''
# client = ZhipuAI(api_key = "09e5da4addba0a593c33a3bf0992b01a.rsVrVkwiB5XZ8Ze5")
client = ZhipuAI(api_key = "5f12e86015e0a1cfe113f50960bfcf0c.T6iNWxjOaztQXpeQ")
MODEL = "glm-4-0520"
#########################################################################################################
# 全部标签的测试
# df_all = pd.read_csv(DATA_PATH)
# assert not df_all.isnull().values.any()

# df_case = df_all.sample(n=1, random_state=42)
# df_test_all = df_all.drop(df_case.index)

# df_test_X = df_test_all.drop(columns=['passed'])
# df_test_y = df_test_all['passed']

# case_instance = df_case.to_dict(orient='records') # 例子字典
# test_x_instance = df_test_X.to_dict(orient='records') #测试无标签字典

# 统一测试集（20%）
df_test_all = pd.read_csv(DATA_PATH_TEST)
x_test = df_test_all.drop(columns=['passed'])
y_test = df_test_all['passed']
df_x_select = x_test[["failures","goout","higher","age","Medu","Fedu","studytime","paid","schoolsup"]]
assert not df_x_select.isnull().values.any()
x_test_instance = df_x_select.to_dict(orient='records')
#############################################################################################################
system_messsage_1 = """You have ample time to carefully understand the varying degrees of influence of the following factors that determine whether a student will pass the final mathematics exam.

                    "age": "student's age (numeric: from 15 to 22)",
                    "studytime": "weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)",
                    "Medu": "mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
                    "higher": "wants to take higher education (binary: yes or no)",
                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)",
                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)",
                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)",
                    "schoolsup": "extra educational support (binary: yes or no)".
                    Analyze the impact of various factors on students' ability to pass mathematics exams. 

                    Based on the following principles, make predictions regarding students' likelihood of passing the exam:
                    1.Goout: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.
                    2.Higher: When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.
                    3.Age:
                    Students aged 21 or 22 will definitely fail the exam.
                    Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Students aged 18 or 20 are likely to pass the exam.
                    4.Failures:
                    Students with 0 failures are highly likely to pass the exam.
                    Students with 2 or 3 failures are highly likely to fail the exam.
                    Studytime: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.
                    5.Paid:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
                    Schoolsup:When schoolsup is 'no', it is very likely that the student will pass the exam.
                    6.Medu/Fedu:
                    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
                    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
                    The influence of factors from highest to lowest is as follows: Failures > Goout = Age > Higher > Medu/Fedu > Schoolsup > Paid > Studytime
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """
user_message_1 = f"""You have ample time to think. Adhere to the above principles, 

    make a judgment on whether the student (failures:3  goout:1 higher:no  age:17  Medu:2  Fedu:1  studytime:1 paid:no schoolsup:yes ) with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    1. Evaluate the Goout attribute.
    2. Evaluate the Higher attribute.
    3. Evaluate the Age attribute.
    4. Evaluate the Failures attribute.
    5. Evaluate the Studytime attribute.
    6. Evaluate the Paid attribute.
    7. Evaluate the Schoolsup attribute.
    8. Evaluate the Medu attributes and Fedu attributes.
    """
assisst_1 = f"""
                    Let's evaluate the student's likelihood of passing the math exam step by step based on the given principles:

                    1. Evaluate the Goout Attribute
                    Goout: 1
                    Principle: Students with goout values of 1, 2, or 3 are highly likely to pass the exam.
                    Judgment: Highly likely to pass.
                    2. Evaluate the Higher Attribute
                    Higher: no
                    Principle: When higher is 'no', it is very likely that the student will fail the exam.
                    Judgment: Very likely to fail.
                    3. Evaluate the Age Attribute
                    Age: 17
                    Principle: Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Judgment: Highly likely to pass.
                    4. Evaluate the Failures Attribute
                    Failures: 3
                    Principle: Students with 2 or 3 failures are highly likely to fail the exam.
                    Judgment: Highly likely to fail.
                    5. Evaluate the Studytime Attribute
                    Studytime: 1
                    Principle: Students who study for 3 or 4 (hours) are highly likely to pass the exam, but students studying for less than 3 (hours) are not guaranteed to pass.
                    Judgment: Not guaranteed to pass.
                    6. Evaluate the Paid Attribute
                    Paid: no
                    Principle: When Paid is 'no', it is likely that the student will fail the exam.
                    Judgment: Likely to fail.
                    7. Evaluate the Schoolsup Attribute
                    Schoolsup: yes
                    Principle: When schoolsup is 'no', it is very likely that the student will pass the exam.
                    Judgment: No specific prediction provided, but implied less likely to pass.
                    8. Evaluate the Medu and Fedu Attributes
                    Medu: 2
                    No specific prediction provided for Medu = 2.
                    Fedu: 1
                    No specific prediction provided for Fedu = 1.
                    Judgment: No direct impact based on provided principles.
                    Final Judgment
                    Considering all the factors and their impacts:

                    Highly likely to pass factors: Goout (1), Age (17)
                    Highly likely to fail factors: Higher (no), Failures (3)
                    Likely to fail factors: Paid (no)
                    Studytime: Not guaranteed to pass
                    Schoolsup: Implied less likely to pass
                    Given the influence ranking:

                    Failures (highly likely to fail)
                    Goout = Age (both highly likely to pass)
                    Higher (very likely to fail)
                    Paid (likely to fail)
                    Studytime (not guaranteed to pass)
                    Schoolsup (no specific prediction)
                    Medu/Fedu (no specific prediction)
                    The factors indicating failure (Failures, Higher, Paid) outweigh those indicating success (Goout, Age). Therefore, the overall judgment is that the student is very likely to fail the exam.

                    Answer: no
            """
#######################################################################################################
system_messsage_2 = f""""""
user_message_2 = f""""""
assisst_3 = f""""""
########################################################################################################
system_messsage_3 = f""""""
user_message_3 = f""""""
assisst_3 = f""""""
#########################################################################################################
system_messsage_4 = f"""                    """
user_message_4 = f""""""
assisst_4 = f""""""
########################################################################################################
system_messsage_5 = f""""""
user_message_5 = f""""""
assisst_5 = f""""""         
########################################################################################################

# prompt1:样例回答用于之后的信息模板。
# response_1 = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#     {"role": "system", "content": system_messsage_1},
#     {"role": "user", "content": user_message_1},
#     {"role": "system", "content": system_messsage_2},
#     {"role": "user", "content": user_message_2},
#     {"role": "system", "content": system_messsage_3},
#     {"role": "user", "content": user_message_3},
#     {"role": "system", "content": system_messsage_4},
#     {"role": "user", "content": user_message_4}
#   ],
#     temperature=0,
# )
# response_case = response_1.choices[0].message.content
# print("response_1_content:", response_case)

messages_template = [ {"role": "system", "content": system_messsage_1},
                      {"role": "user", "content": user_message_1},
                      {"role":"assistant","content": assisst_1},]
                    #  {"role": "system", "content": system_messsage_2},
                    #  {"role": "user", "content": user_message_2},
                    #  {"role":"assistant","content": assisst_2},
                    #  {"role": "system", "content": system_messsage_3},
                    #  {"role": "user", "content": user_message_3},
                    #  {"role":"assistant","content": assisst_3},
                    #  {"role": "system", "content": system_messsage_4},
                    #  {"role": "user", "content": user_message_4},
                    #  {"role":"assistant","content": assisst_4},
                    #  {"role": "system", "content": system_messsage_5},
                    #  {"role": "user", "content": user_message_5},
                    #  {"role":"assistant","content": assisst_5},
                    


# 提问的system_message
system_message = """You have ample time to carefully understand the varying degrees of influence of the following factors that determine whether a student will pass the final mathematics exam.

                    "age": "student's age (numeric: from 15 to 22)",
                    "studytime": "weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)",
                    "Medu": "mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
                    "higher": "wants to take higher education (binary: yes or no)",
                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)",
                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)",
                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)",
                    "schoolsup": "extra educational support (binary: yes or no)".
                    Analyze the impact of various factors on students' ability to pass mathematics exams. 

                    Based on the following principles, make predictions regarding students' likelihood of passing the exam:
                    1.Goout: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.
                    2.Higher: When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.
                    3.Age:
                    Students aged 21 or 22 will definitely fail the exam.
                    Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Students aged 18 or 20 are likely to pass the exam.
                    4.Failures:
                    Students with 0 failures are highly likely to pass the exam.
                    Students with 2 or 3 failures are highly likely to fail the exam.
                    Studytime: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.
                    5.Paid:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
                    Schoolsup:When schoolsup is 'no', it is very likely that the student will pass the exam.
                    6.Medu/Fedu:
                    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
                    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """

responses_select_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have ample time to think. Adhere to the above principles, 

    make a judgment on whether the student {testcase} with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    1. Evaluate the Goout attribute.
    2. Evaluate the Higher attribute.
    3. Evaluate the Age attribute.
    4. Evaluate the Failures attribute.
    5. Evaluate the Studytime attribute.
    6. Evaluate the Paid attribute.
    7. Evaluate the Schoolsup attribute.
    8. Evaluate the Medu attributes and Fedu attributes.
    
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """
    # responses_list = []
    # for _ in range(5):
    #     response_1 = client.chat.completions.create(
    #         model=MODEL,
    #         messages =messages_template + [{"role":"system","content":system_message}] + [{"role": "user","content":question_prompt}],
    #         stream = False,
    #         temperature=0.5,

    #     )
    #     response_test = response_1.choices[0].message.content
    #     responses_list.append(response_test) 
    #     print(" response_test:", response_test)
    #     print("responses_list:",responses_list)
    # response_2 = client.chat.completions.create(
    #         model=MODEL,
    #         messages = [{"role":"system","content":"You will be given 5 answers to the same question, please choose the answer that appears most often. There are only two possible answers, yes or no."}] + [{"role": "user","content":f"Please select the answer that appears most often from the 5 answers in {responses_list} and answer yes or no."}],
    #         stream = False,
    #         temperature=0,
    #     )
    # response_select = response_2.choices[0].message.content
    # responses_select_list.append(response_select)
    # print("response_select:",response_select)
    # print("responses_select_list:",responses_select_list)


# 数据可视化
y_pred = ['No', 'yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'no', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 
'Yes', 'Yes', 'Yes', 'yes', 'No']
y_pred = np.char.lower(y_pred)
y_pred_numeric = np.where(y_pred == 'yes', 1, 0)
print("y_pred", y_pred_numeric)
 
y_true = y_test
y_true = np.array(y_true, dtype=str)
y_true = np.char.lower(y_true)
y_true_numeric = np.where(y_true == 'yes', 1, 0)
print("y_test", y_true_numeric)

mse = mean_squared_error( y_true_numeric, y_pred_numeric)
rmse = np.sqrt(mse)

print(classification_report(y_true_numeric, y_pred_numeric, digits=3))
print("RMSE为：", rmse)

# 计算混淆矩阵
cf_matrix = confusion_matrix(
    y_true,
    y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labs = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labs = np.asarray(labs).reshape(2,2)
plt.figure(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=labs, fmt='', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#(The degree words used in the principles, the order of degree from high to low is: definitely, highly likely, very likely, likely)