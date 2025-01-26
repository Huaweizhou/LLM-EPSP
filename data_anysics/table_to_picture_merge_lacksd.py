import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置 Seaborn 风格
sns.set(style="white", palette="bright", font_scale=1.2)
save_dir = "picture_experiment"
os.makedirs(save_dir, exist_ok=True)

# 数据整理（与之前相同）
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

# 将数据转化为 DataFrame（与之前相同）
rows = []
for split, methods in data.items():
    for method, metrics in methods.items():
        rows.append({"Split": split, "Method": method, **metrics})

df = pd.DataFrame(rows)

# 定义颜色映射（与之前相同，但建议使用更易区分的调色板）
methods = df['Method'].unique()
# 建议使用例如 "husl", "tab10", "Set1" 等更易区分的调色板
palette = sns.color_palette("husl", len(methods))  # 修改这里
color_map = dict(zip(methods, palette))

# 指标列表
metrics = ["Acc.", "Prec.", "Rec.", "F1", "RMSE"]

# 创建子图
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        data=df,
        x="Split",
        y=metric,
        hue="Method",
        palette=color_map,
        ax=ax
    )
    ax.set_title(f"{metric} Comparison")
    ax.set_xlabel("Split Ratio")
    ax.set_ylabel(metric)
    ax.get_legend().remove()  # 移除每个子图的图例

# 获取图例句柄和标签（只需获取一次）
handles, labels = axes[0].get_legend_handles_labels()

# 创建主图例，并调整布局使其完整显示
fig.legend(handles, labels, title="", loc="upper center", ncol=len(methods)//2+len(methods)%2, bbox_to_anchor=(0.5, 1.15)) # ncol修改这里，bbox_to_anchor修改这里

plt.tight_layout(rect=[0, 0, 1, 1])  # 调整rect参数，为图例留出足够的空间
plt.savefig(os.path.join(save_dir, "combined_metrics_comparison.png"), dpi=300)
plt.show()