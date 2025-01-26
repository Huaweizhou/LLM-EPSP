import pandas as pd

df = pd.read_csv('E:\zhw_git\students-performance-prediction-UCI\dataset\dadaset_OULAD\studentInfo_AAA_2014J.csv')

gender_counts = df['gender'].value_counts()
print("gender_counts",gender_counts)

region_counts = df['region'].value_counts()
print("region_counts",region_counts)

highest_education_counts = df['highest_education'].value_counts()
print("highest_education_counts",highest_education_counts)

imd_band_counts = df['imd_band'].value_counts()
print("imd_band_counts",imd_band_counts)

age_band_counts = df['age_band'].value_counts()
print("age_band_counts",age_band_counts)

num_of_prev_attempts_counts = df['num_of_prev_attempts'].value_counts()
print("num_of_prev_attempts_counts",num_of_prev_attempts_counts)

studied_credits_counts = df['studied_credits'].value_counts()
print("studied_credits_counts",studied_credits_counts)

disability_counts = df['disability'].value_counts()
print("disability_counts",disability_counts)


# 将每个标签的独立值转换为从 0 开始的数字标记
gender_labels, gender_mapping = pd.factorize(df['gender'])
df['gender'] = gender_labels
print("gender_mapping:", dict(enumerate(gender_mapping)))

region_labels, region_mapping = pd.factorize(df['region'])
df['region'] = region_labels
print("region_mapping:", dict(enumerate(region_mapping)))

highest_education_labels, highest_education_mapping = pd.factorize(df['highest_education'])
df['highest_education'] = highest_education_labels
print("highest_education_mapping:", dict(enumerate(highest_education_mapping)))

imd_band_labels, imd_band_mapping = pd.factorize(df['imd_band'])
df['imd_band'] = imd_band_labels
print("imd_band_mapping:", dict(enumerate(imd_band_mapping)))

age_band_labels, age_band_mapping = pd.factorize(df['age_band'])
df['age_band'] = age_band_labels
print("age_band_mapping:", dict(enumerate(age_band_mapping)))

num_of_prev_attempts_labels, num_of_prev_attempts_mapping = pd.factorize(df['num_of_prev_attempts'])
df['num_of_prev_attempts'] = num_of_prev_attempts_labels
print("num_of_prev_attempts_mapping:", dict(enumerate(num_of_prev_attempts_mapping)))

studied_credits_labels, studied_credits_mapping = pd.factorize(df['studied_credits'])
df['studied_credits'] = studied_credits_labels
print("studied_credits_mapping:", dict(enumerate(studied_credits_mapping)))

disability_labels, disability_mapping = pd.factorize(df['disability'])
df['disability'] = disability_labels
print("disability_mapping:", dict(enumerate(disability_mapping)))
