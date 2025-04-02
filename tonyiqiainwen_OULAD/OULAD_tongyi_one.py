import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from zhipuai import ZhipuAI
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, mean_squared_error

DATA_PATH = "dataset\dadaset_OULAD\studentInfo_AAA_2014J_trian.csv"
DATA_PATH_TEST = "dataset\dadaset_OULAD\studentInfo_AAA_2014J_test.csv"

IN_COLS = [
    "code_module","code_presentation","gender","region","highest_education"\
    "imd_band","age_band","num_of_prev_attempts","studied_credits","disability","final_result"]

#########################################################################################################
'''
api相关的属性
'''
api_key = "sk-5735d37f3be1495f8261dea4a75c36ae"
api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=api_key, base_url=api_base)
MODEL = "qwen-max"
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
system_messsage_1 =  """You have sufficient time to carefully evaluate the influence of the following factors on whether a student will pass the final exam. Each factor varies in its degree of impact, and your task is to analyze these differences.

                    Gender: The gender of the student (categorical: M for Male, F for Female).
                    Region: The geographic region where the student resided during the module presentation (categorical: 'East Anglian Region', 'Scotland', 'South East Region', 'West Midlands Region', 'Wales', 'North Western Region', 'North Region', 'South Region', 'Ireland', 'South West Region', 'East Midlands Region', 'Yorkshire Region', 'London Region').
                    Highest Education: The highest level of education the student had upon entering the module presentation (categorical: 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification').
                    IMD Band: The Index of Multiple Deprivation (IMD) band for the area where the student lived during the module (numeric: '80-90%', '60-70%', '40-50%', '30-40%', '0-10%', '90-100%', '70-80%', '10-20%', '50-60%', '20-30%').
                    Age Band: The student's age group (categorical:'0-35', '35-55','55<=').
                    Number of Previous Attempts: The number of times the student has previously attempted this module (numeric: '0', '1').
                    Studied Credits: The total number of credits the student is currently enrolled in (numeric: 60, 90, 120, 150, 180, 210, 240, 300).
                    Disability: Whether the student has declared a disability (categorical: 'N' for No, 'Y' for Yes).
                    Please carefully analyze how these factors may influence a student's likelihood of passing the final exam. After considering all the factors thoroughly, confirm your understanding of their potential impact.
                """
user_message_1 = f"""You have sufficient time to carefully consider. 
    
Make a judgment on whether the student （gender：F,region:London Region,highest_education:A Level or Equivalent,imd_band:20-30%,   age_band:0-35,num_of_prev_attempts:0,studied_credits:60,disability:N) with the given attributes can pass the math exam.

If you believe the student can pass the exam, respond with 'yes'. If you believe the student cannot pass, respond with 'no'.

Provide only 'yes' or 'no' as your final response—no additional explanation or analysis.
                  """
assisst_1 = f"""
                Student Attributes
                Gender: Female (F)
                Region: London Region
                Highest Education: A Level or Equivalent
                IMD Band: 20-30%
                Age Band: 0-35
                Number of Previous Attempts: 0
                Studied Credits: 60
                Disability: No (N)
                Analysis of Each Attribute
                Gender:

                Gender alone is not a strong predictor of academic success; however, studies often show nuanced outcomes based on context.
                Region (London Region):

                London generally has better educational resources and support systems, which can positively impact student performance.
                Highest Education (A Level or Equivalent):

                Having A Level qualifications suggests that the student has a solid foundation in relevant subjects, enhancing the likelihood of passing the math exam.
                IMD Band (20-30%):

                An IMD band of 20-30% indicates that the student is from a relatively disadvantaged background, but this does not negate their potential to succeed, especially with other supportive factors.
                Age Band (0-35):

                Being within the 0-35 age band generally indicates that the student is likely to be more engaged and familiar with current educational practices.
                Number of Previous Attempts (0):

                Having no previous attempts suggests that this is the student's first try, which could mean they are fresh and motivated, possibly leading to a positive outcome.
                Studied Credits (60):

                While 60 credits indicate a manageable workload, it also suggests the student is actively engaged in their studies, which is typically linked to better performance.
                Disability (No):

                Not having a declared disability means there are fewer external challenges that could impede academic success.
                Conclusion
                Considering the combination of these factors:

                The student has foundational qualifications (A Level), is in a supportive region (London), and has not previously attempted the module, which generally leads to a positive outlook on their potential to pass the exam.
                The supportive aspects (region and education) outweigh the challenges posed by the IMD band.
                Given this analysis, the conclusion is that the student is likely to pass the math exam.

                Final response: yes.
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
                """

responses_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have sufficient time to carefully consider. 
    
Make a judgment on whether the student {testcase} with the given attributes can pass the math exam.

If you believe the student can pass the exam, respond with 'yes'. If you believe the student cannot pass, respond with 'no'.

Provide only 'yes' or 'no' as your final response—no additional explanation or analysis.
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
    time.sleep(1)


# 数据可视化
y_pred = responses_list
# y_pred = ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
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

print(classification_report(y_true_numeric, y_pred_numeric, digits=3))
print("RMSE为：", rmse)

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