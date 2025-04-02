import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('student-data.csv')

# 获取总行数
total_rows = len(df)

# 计算需要抽取的行数（约20%）
sample_size = int(total_rows * 0.2)

# 使用sample方法随机抽取约20%的数据
sampled_df = df.sample(n=sample_size, random_state=42)
remaining_df = df.loc[~df.index.isin(sampled_df.index)]

# 将抽样结果保存到新的CSV文件（可选）
sampled_df.to_csv('student-data-test.csv', index=False)
remaining_df.to_csv('student-data-train-valid.csv', index=False)

print(f"原始数据行数: {total_rows}")
print(f"抽样后数据行数: {len(sampled_df)}")
print(f"抽样比例: {len(sampled_df) / total_rows:.2%}")