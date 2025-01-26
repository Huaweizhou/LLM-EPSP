import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置 Seaborn 风格
# sns.set(style="whitegrid", palette="muted", font_scale=1.2)
sns.set(style="white", palette="deep", font_scale=1.2)
save_dir = "picture_experiment"
os.makedirs(save_dir, exist_ok=True) 
# 数据整理：以嵌套字典存储表格数据
data = {
    "7:3": {
        "LR": {"Acc.": 0.85, "Prec.": 0.82, "Rec.": 0.83, "F1": 0.82, "RMSE": 0.17},
        "KNN": {"Acc.": 0.87, "Prec.": 0.84, "Rec.": 0.83, "F1": 0.82, "RMSE": 0.15},
        "SVM": {"Acc.": 0.88, "Prec.": 0.85, "Rec.": 0.87, "F1": 0.86, "RMSE": 0.14},
        "NB": {"Acc.": 0.82, "Prec.": 0.79, "Rec.": 0.80, "F1": 0.79, "RMSE": 0.18},
        "DT": {"Acc.": 0.86, "Prec.": 0.84, "Rec.": 0.85, "F1": 0.84, "RMSE": 0.16},
        "RF": {"Acc.": 0.92, "Prec.": 0.91, "Rec.": 0.91, "F1": 0.91, "RMSE": 0.11},
        "MLP": {"Acc.": 0.89, "Prec.": 0.87, "Rec.": 0.88, "F1": 0.88, "RMSE": 0.13},
        "GBC": {"Acc.": 0.91, "Prec.": 0.89, "Rec.": 0.89, "F1": 0.88, "RMSE": 0.11},
        "ACO-DT": {"Acc.": 0.90, "Prec.": 0.88, "Rec.": 0.89, "F1": 0.88, "RMSE": 0.12},
        "ANFIS": {"Acc.": 0.91, "Prec.": 0.89, "Rec.": 0.90, "F1": 0.89, "RMSE": 0.12}
    },
    "8:2": {
        "LR": {"Acc.": 0.85, "Prec.": 0.82, "Rec.": 0.83, "F1": 0.82, "RMSE": 0.17},
        "KNN": {"Acc.": 0.87, "Prec.": 0.84, "Rec.": 0.83, "F1": 0.82, "RMSE": 0.15},
        "SVM": {"Acc.": 0.88, "Prec.": 0.85, "Rec.": 0.87, "F1": 0.86, "RMSE": 0.14},
        "NB": {"Acc.": 0.82, "Prec.": 0.79, "Rec.": 0.80, "F1": 0.79, "RMSE": 0.18},
        "DT": {"Acc.": 0.86, "Prec.": 0.84, "Rec.": 0.85, "F1": 0.84, "RMSE": 0.16},
        "RF": {"Acc.": 0.92, "Prec.": 0.91, "Rec.": 0.91, "F1": 0.91, "RMSE": 0.11},
        "MLP": {"Acc.": 0.89, "Prec.": 0.87, "Rec.": 0.88, "F1": 0.88, "RMSE": 0.13},
        "GBC": {"Acc.": 0.91, "Prec.": 0.89, "Rec.": 0.89, "F1": 0.88, "RMSE": 0.11},
        "ACO-DT": {"Acc.": 0.90, "Prec.": 0.88, "Rec.": 0.89, "F1": 0.88, "RMSE": 0.12},
        "ANFIS": {"Acc.": 0.91, "Prec.": 0.89, "Rec.": 0.90, "F1": 0.89, "RMSE": 0.12}
    }, # 数据同上，简略省略
    "5:5": {
        "LR": {"Acc.": 0.85, "Prec.": 0.82, "Rec.": 0.83, "F1": 0.82, "RMSE": 0.17},
        "KNN": {"Acc.": 0.87, "Prec.": 0.84, "Rec.": 0.83, "F1": 0.82, "RMSE": 0.15},
        "SVM": {"Acc.": 0.88, "Prec.": 0.85, "Rec.": 0.87, "F1": 0.86, "RMSE": 0.14},
        "NB": {"Acc.": 0.82, "Prec.": 0.79, "Rec.": 0.80, "F1": 0.79, "RMSE": 0.18},
        "DT": {"Acc.": 0.86, "Prec.": 0.84, "Rec.": 0.85, "F1": 0.84, "RMSE": 0.16},
        "RF": {"Acc.": 0.92, "Prec.": 0.91, "Rec.": 0.91, "F1": 0.91, "RMSE": 0.11},
        "MLP": {"Acc.": 0.89, "Prec.": 0.87, "Rec.": 0.88, "F1": 0.88, "RMSE": 0.13},
        "GBC": {"Acc.": 0.91, "Prec.": 0.89, "Rec.": 0.89, "F1": 0.88, "RMSE": 0.11},
        "ACO-DT": {"Acc.": 0.90, "Prec.": 0.88, "Rec.": 0.89, "F1": 0.88, "RMSE": 0.12},
        "ANFIS": {"Acc.": 0.91, "Prec.": 0.89, "Rec.": 0.90, "F1": 0.89, "RMSE": 0.12}
    } # 数据同上，简略省略
}

# 将数据转化为 DataFrame
rows = []
for split, methods in data.items():
    for method, metrics in methods.items():
        rows.append({"Split": split, "Method": method, **metrics})

df = pd.DataFrame(rows)

# 定义颜色映射
methods = df['Method'].unique()
palette = sns.color_palette("husl", len(methods))
color_map = dict(zip(methods, palette))

# 指标列表
metrics = ["Acc.", "Prec.", "Rec.", "F1", "RMSE"]

# 绘制每个指标的图表
for metric in metrics:
    plt.figure(figsize=(12, 6))
    
    # 使用 Seaborn 绘制分组柱状图
    sns.barplot(
        data=df,
        x="Split",  # 划分比例
        y=metric,   # 当前指标
        hue="Method",  # 不同方法
        palette=color_map
    )
    
    # 设置标题与标签
    plt.title(f"Comparison of {metric} for Different Models")
    plt.xlabel("Split Ratio")
    plt.ylabel(metric)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric}_comparison.png"), dpi=300)
plt.show()
