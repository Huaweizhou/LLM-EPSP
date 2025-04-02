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

DATA_PATH = "dataset\dadaset_OULAD\studentInfo_AAA_2014J_trian.csv"
DATA_PATH_TEST = "dataset\dadaset_OULAD\studentInfo_AAA_2014J_test.csv"

IN_COLS = [
    "code_module","code_presentation","id_stu5f12e86015e0a1cfe113f50960bfcf0c.T6iNWxjOaztQXpeQdent","gender","region","highest_education"\
    "imd_band","age_band","num_of_prev_attempts","studied_credits","disability","final_result"]

#########################################################################################################
'''
api相关的属性
'''
# client = ZhipuAI(api_key = "5f12e86015e0a1cfe113f50960bfcf0c.T6iNWxjOaztQXpeQ")
# client = ZhipuAI(api_key = "09e5da4addba0a593c33a3bf0992b01a.rsVrVkwiB5XZ8Ze5")
client = ZhipuAI(api_key = "4d177fa7b97da9d87007acf21b11f473.iqDWF4ZtKvs3JO1a")
# MODEL = "glm-4-0520"
MODEL = "glm-4-plus"
#########################################################################################################

# 统一测试集（20%）
df_test_all = pd.read_csv(DATA_PATH_TEST)
df_test_all.dropna(inplace=True)
x_test = df_test_all.drop(columns=['final_result'])
y_test = df_test_all['final_result']
df_x_select = x_test[["gender","region","highest_education","imd_band","age_band","num_of_prev_attempts","studied_credits","disability"]]
assert not df_x_select.isnull().values.any()
x_test_instance = df_x_select.to_dict(orient='records')
#############################################################################################################
system_messsage_1 =  """
                """

user_message_1 = f"""
    
                  """
assisst_1 = f"""
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
system_message = """You have sufficient time to carefully evaluate the influence of the following factors on whether a student will pass the final exam. Each factor varies in its degree of impact, and your task is to analyze these differences.

                    Gender: The gender of the student (categorical: M for Male, F for Female).
                    Region: The geographic region where the student resided during the module presentation (categorical: 'East Anglian Region', 'Scotland', 'South East Region', 'West Midlands Region', 'Wales', 'North Western Region', 'North Region', 'South Region', 'Ireland', 'South West Region', 'East Midlands Region', 'Yorkshire Region', 'London Region').
                    Highest Education: The highest level of education the student had upon entering the module presentation (categorical: 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification').
                    IMD Band: The Index of Multiple Deprivation (IMD) band for the area where the student lived during the module (numeric: '80-90%', '60-70%', '40-50%', '30-40%', '0-10%', '90-100%', '70-80%', '10-20%', '50-60%', '20-30%').
                    Age Band: The student's age group (categorical:'0-35', '35-55','55<=').
                    Number of Previous Attempts: The number of times the student has previously attempted this module (numeric: '0', '1').
                    Studied Credits: The total number of credits the student is currently enrolled in (numeric: 60, 90, 120, 150, 180, 210, 240, 300).
                    Disability: Whether the student has declared a disability (categorical: 'N' for No, 'Y' for Yes).
                    Please carefully analyze how these factors may influence a student's likelihood of passing the final exam. After considering all the factors thoroughly, confirm your understanding of their potential impact.

                    Based on the following principles, make predictions regarding students' likelihood of passing the exam:
                    Gender: The student's gender has little to no influence on their likelihood of passing.
                    Region:
                    Students from the North Region, South East Region, or Yorkshire Region are highly likely to pass.
                    Students from the South Region or West Midlands Region are likely to pass.
                    Students from Wales are highly unlikely to pass.
                    Highest Education: The student’s highest level of education has minimal influence on their likelihood of passing.
                    IMD Band:
                    Students in the 10-20 or 60-70 bands are highly unlikely to pass.
                    Students in the 80-90 band are highly likely to pass.
                    Students in the 40-50 or 90-100 bands are likely to pass.
                    Age Band: The student’s age has little to no influence on their likelihood of passing.
                    Number of Previous Attempts:
                    Students with 0 previous attempts are slightly more likely to pass.
                    Studied Credits:
                    Students with 150 or 300 credits cannot pass.
                    Students with 240 or 210 credits will definitely pass.
                    Students with 60, 90, or 120 credits are highly likely to pass.
                    Students with 180 credits are unlikely to pass.
                    Disability: Non-disabled students are slightly more likely to pass compared to students with a disability.
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """

responses_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have sufficient time to carefully consider. 
    
Make a judgment on whether the student {testcase} with the given attributes can pass the math exam.

If you believe the student can pass the exam, respond with 'yes'. If you believe the student cannot pass, respond with 'no'.

Provide only 'yes' or 'no' as your final response—no additional explanation or analysis.

Let's work this out in a step by step way to be sure we have the right answer.
    1. Evaluate the gender attribute.
    2. Evaluate the region attribute.
    3. Evaluate the highest_education attribute.
    4. Evaluate the imd_band attribute.
    5. Evaluate the age_band attribute.
    6. Evaluate the num_of_prev_attempts attribute.
    7. Evaluate the studied_credits attribute.
    8. Evaluate the disability attribute.
    
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """

    # response_2 = client.chat.completions.create(
    #     model=MODEL,
    #     messages = [{"role":"system","content":system_message}] + [{"role": "user","content":question_prompt}],
    #     stream = False,
    #     temperature=0,
    # )
    # response_test = response_2.choices[0].message.content
    # responses_list.append(response_test)
    # print("response_test:", response_test)
    # print("responses_list:", responses_list)

# 数据可视化

y_pred = responses_list
# zero就将1，2，3，4，5，6删除
# y_pred = ['yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no']
# zeroSystem_message = 存在着限制信息时 0.759
y_pred = ['yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no']
# System_message = 不存在限制信息时 0.722
# y_pred = ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']
y_pred = np.char.lower(y_pred)
y_pred_numeric = np.where(y_pred == 'yes', 1, 0)
print("y_pred", y_pred_numeric)
 
y_true = y_test
y_true = np.array(y_true, dtype=str)
y_true = np.char.lower(y_true)
y_true_numeric = np.where(y_true == 'pass', 1, 0)
print("y_test", y_true_numeric)

mse = mean_squared_error( y_true_numeric, y_pred_numeric)
rmse = np.sqrt(mse)
auc = roc_auc_score(y_true_numeric, y_pred_numeric)
print(classification_report(y_true_numeric, y_pred_numeric, digits=3))
print("RMSE为：", rmse)
print("AUC为：", auc)
# 计算混淆矩阵
cf_matrix = confusion_matrix(
    y_true_numeric,
    y_pred_numeric)
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