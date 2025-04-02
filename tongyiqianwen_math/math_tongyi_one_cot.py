import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
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
api_key = "sk-5735d37f3be1495f8261dea4a75c36ae"
api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "qwen-max"
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
system_messsage_1 =  """You have ample time to carefully understand the varying degrees of influence of the following factors that determine whether a student will pass the final mathematics exam.

                    "age": "student's age (numeric: from 15 to 22)",

                    "studytime": "weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)",

                    "Medu": "mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",

                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 
                    3 - secondary education or 4 - higher education)",

                    "higher": "wants to take higher education (binary: yes or no)",

                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)",

                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)",

                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)",

                    "schoolsup": "extra educational support (binary: yes or no)".

                    Analyze the impact of various factors on students' ability to pass mathematics exams. 

                    Strictly based on the following principles, make predictions regarding students' likelihood of passing the exam:

                    1.Goout: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.

                    2.Higher: When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.

                    3.Age:
                    Students aged 21 or 22 will definitely fail the exam.
                    Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Students aged 18 or 20 are likely to pass the exam.

                    4.Failures:
                    Students with 0 failures are highly likely to pass the exam.
                    Students with 2 or 3 failures are highly likely to fail the exam.

                    5.Studytime: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.

                    6.Paid:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
                    
                    7.Schoolsup:When schoolsup is 'no', it is very likely that the student will pass the exam.

                    8.Medu/Fedu:
                    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
                    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """
user_message_1 = f"""
                   You have ample time to think.You have ample time to think. Strictly adhere to the above principles, 

                   make a judgment on whether the student(failures:1,goout:4,higher:yes,age:16,Medu:1,Fedu:2,studytime:1,paid:no,schoolsup:yes,pass:no)with the following attributes can pass the math exam.
                   (Pay special attention to degree words)

                   If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
                   Let's work this out in a step by step way to be sure we have the right answer.
    
                  """
assisst_1 = f"""
                    Let's analyze each factor for the given student to make an informed judgment on whether they can pass the math exam:

                    Failures: 1
                    The student has one past class failure. This indicates some difficulty with academic performance but not an insurmountable barrier. This factor suggests a need for improvement but does not decisively indicate failure.
                    Goout: 4
                    The student frequently goes out with friends (4 on a scale of 1 to 5). High social activity can detract from study time and focus, which may negatively impact academic performance.
                    Higher: Yes
                    The student wants to pursue higher education. This motivation can drive the student to work harder and improve their performance, which is a positive factor.
                    Age: 16
                    The student is 16 years old. This is relatively young and could imply less maturity, but age alone is not a definitive factor in academic success.
                    Medu: 1
                    The mother has primary education (4th grade). This may limit the academic support available at home, potentially a negative factor.
                    Fedu: 2
                    The father has education up to 5th to 9th grade. This provides slightly better academic support than the mother's education level but still limited.
                    Studytime: 1
                    The student studies less than 2 hours per week. This is a significant negative factor, as insufficient study time is likely to result in poor understanding and retention of material.
                    Paid: No
                    The student does not receive extra paid classes in Math. This removes a potential source of additional academic support.
                    Schoolsup: Yes
                    The student receives extra educational support. This is a positive factor, as it can help address learning gaps and provide additional resources for improvement.
                    Pass: No (current status)
                    The student has not passed the exam up to this point.
                    Step-by-Step Analysis
                    Motivation (Higher: Yes): This is a strong positive factor, indicating the student's desire to improve and succeed.
                    Study Habits (Studytime: 1): This is a major negative factor, as less than 2 hours of study per week is far below what is generally needed to perform well in exams.
                    Social Activity (Goout: 4): High social activity can be detrimental to academic focus and study time.
                    Parental Education (Medu: 1, Fedu: 2): Limited educational background of parents may mean less academic support at home.
                    Past Performance (Failures: 1): Indicates some difficulty but not a definitive predictor of future failure.
                    Additional Support (Schoolsup: Yes): Positive factor providing extra help.
                    Extra Classes (Paid: No): Lack of additional tutoring is a negative factor.
                    Judgment
                    Given the critical negative factor of insufficient study time (Studytime: 1) and the high level of social activity (Goout: 4), it seems unlikely that the student will be able to pass the math exam without significant changes to their study habits and focus. While the desire to pursue higher education and the extra educational support are positive indicators, they may not be enough to overcome the lack of adequate study time.

                    Answer: No

                    The student, based on the current attributes and behaviors, is unlikely to pass the math exam.
            """
#######################################################################################################
system_messsage_2 = f"""With ample time to think, based on your understanding of the factors influencing whether a student will pass the math exam, make a judgment."""
user_message_2 = f""""""
assisst_3 = f""""""
########################################################################################################
system_messsage_3 = f"""With ample time to think, based on your understanding of the factors influencing whether a student will pass the math exam, make a judgment."""
user_message_3 = f""""""
assisst_3 = f""""""
#########################################################################################################
system_messsage_4 = f"""With ample time to think, based on your understanding of the factors influencing whether a student will pass the math exam, make a judgment.                    """
user_message_4 = f""""""
assisst_4 = f""""""
########################################################################################################
system_messsage_5 = f"""With ample time to think, based on your understanding of the factors influencing whether a student will pass the math exam, make a judgment."""
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

                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 
                    3 - secondary education or 4 - higher education)",

                    "higher": "wants to take higher education (binary: yes or no)",

                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)",

                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)",

                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)",

                    "schoolsup": "extra educational support (binary: yes or no)".

                    Analyze the impact of various factors on students' ability to pass mathematics exams. 

                    Strictly based on the following principles, make predictions regarding students' likelihood of passing the exam:

                    1.Goout: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.

                    2.Higher: When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.

                    3.Age:
                    Students aged 21 or 22 will definitely fail the exam.
                    Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Students aged 18 or 20 are likely to pass the exam.

                    4.Failures:
                    Students with 0 failures are highly likely to pass the exam.
                    Students with 2 or 3 failures are highly likely to fail the exam.

                    5.Studytime: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.

                    6.Paid:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
                    
                    7.Schoolsup:When schoolsup is 'no', it is very likely that the student will pass the exam.

                    8.Medu/Fedu:
                    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
                    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """

responses_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have ample time to think. Strictly adhere to the above principles, 

    make a judgment on whether the student {testcase} with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """
    response_2 = client.chat.completions.create(
        model=MODEL,
        messages =messages_template + [{"role":"system","content":system_message}] + [{"role": "user","content":question_prompt}],
        stream = False,
        temperature=0,
    )
    response_test = response_2.choices[0].message.content
    responses_list.append(response_test)
    print("response_test:", response_test)
    print("responses_list:", responses_list)
    time.sleep(5)
    # # # 流式输出
    # response_test = ""
    # for partial_response in response_2:
    #     response_test += partial_response['choices'][0]['delta'].get('content', '')
    #     response_test contains the full response text
    # print("responses_list",responses_list)


# 数据可视化
y_pred = responses_list
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