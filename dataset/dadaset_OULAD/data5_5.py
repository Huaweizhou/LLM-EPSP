import pandas as pd

# 读取 CSV 文件
df = pd.read_csv(r'E:\zhw_git\students-performance-prediction-UCI\dataset\dadaset_OULAD\studentInfo_AAA.csv')

# 确保 final_result 列中只包含 'Pass' 和 'Fail'
assert set(df['final_result'].unique()).issubset({'Pass', 'Fail'})

# 计算 Pass 和 Fail 的数量
pass_count = df[df['final_result'] == 'Pass'].shape[0]
fail_count = df[df['final_result'] == 'Fail'].shape[0]

# 计算比例
total_count = pass_count + fail_count
pass_ratio = pass_count / total_count
fail_ratio = fail_count / total_count

# 计算需要从每个类别中抽取的样本数量
num_samples = 80
pass_samples = int(num_samples * pass_ratio)
fail_samples = int(num_samples * fail_ratio)

# 分别获取 Pass 和 Fail 的样本
pass_samples_df = df[df['final_result'] == 'Pass']
fail_samples_df = df[df['final_result'] == 'Fail']

# 从每个类别中随机抽样
pass_sampled = pass_samples_df.sample(pass_samples, random_state=42)
fail_sampled = fail_samples_df.sample(fail_samples, random_state=42)

# 合并样本
test_samples = pd.concat([pass_sampled, fail_sampled])

# 剩余的学生作为训练集
train_samples = df.drop(test_samples.index)

# 保存测试集和训练集到 CSV 文件
test_samples.to_csv(r'E:\zhw_git\students-performance-prediction-UCI\dataset\dadaset_OULAD\studentInfo_AAA_Test.csv', index=False)
train_samples.to_csv(r'E:\zhw_git\students-performance-prediction-UCI\dataset\dadaset_OULAD\studentInfo_AAA_Train.csv', index=False)

print("测试集和训练集已保存。")
