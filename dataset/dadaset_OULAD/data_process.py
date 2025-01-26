import pandas as pd
df = pd.read_csv('dataset\dadaset_OULAD\studentInfo.csv')
PF_df = df[df['final_result'].isin(['Pass', 'Fail'])]
PF_df.to_csv('dataset\dadaset_OULAD\studentInfo_PF.csv', index=False)

AAA_df_2013J = PF_df[(PF_df['code_module'] == 'AAA') & (PF_df['code_presentation'] == '2013J')]
AAA_df_2013J.to_csv('dataset\dadaset_OULAD\studentInfo_AAA_2013J.csv', index=False)

AAA_df_2014J = PF_df[(PF_df['code_module'] == 'AAA') & (PF_df['code_presentation'] == '2014J')]
AAA_df_2014J.to_csv('dataset\dadaset_OULAD\studentInfo_AAA_2014J.csv', index=False)

module_counts = PF_df['code_module'].value_counts()
student_counts_AAA2013J = AAA_df_2013J['id_student'].value_counts()
student_counts_AAA2014J = AAA_df_2014J['id_student'].value_counts()

print("module_counts",module_counts)
print("student_counts_AAA2013J",student_counts_AAA2013J)
print("student_counts_AAA2014J",student_counts_AAA2014J)