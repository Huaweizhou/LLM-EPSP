import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 设置 Seaborn 风格
# sns.set_palette(sns.color_palette("viridis"))
sns.set(style="ticks", palette="bright", font_scale=1.2)

save_dir = "picture_experiment"
os.makedirs(save_dir, exist_ok=True)

data = {
    "7:3": {
        "LR": {"Acc.": 0.671, "Prec.": 0.618, "Rec.": 0.563, "F1": 0.550,"RMSE": 0.573},
        "KNN": {"Acc.": 0.658, "Prec.": 0.583, "Rec.": 0.581, "F1": 0.458, "RMSE": 0.584},
        "SVM": {"Acc.": 0.662, "Prec.": 0.589, "Rec.": 0.577, "F1": 0.527, "RMSE": 0.575},
        "NB": {"Acc.": 0.684, "Prec.": 0.648, "Rec.": 0.573, "F1": 0.559,"RMSE": 0.563},
        "DT": {"Acc.": 0.679, "Prec.": 0.613, "Rec.": 0.548, "F1": 0.544, "RMSE": 0.593},
        "RF": {"Acc.": 0.658, "Prec.": 0.599, "Rec.": 0.571, "F1": 0.569, "RMSE": 0.584},
        "MLP": {"Acc.": 0.644, "Prec.": 0.584, "Rec.": 0.588, "F1": 0.587, "RMSE": 0.615},
        "GBC": {"Acc.": 0.684, "Prec.": 0.658, "Rec.": 0.564, "F1": 0.541, "RMSE": 0.562},
        "ACO-DT": {"Acc.": 0.713, "Prec.": 0.679, "Rec.": 0.616, "F1": 0.622, "RMSE": 0.537},
        "ANFIS": {"Acc.": 0.709, "Prec.": 0.667, "Rec.": 0.623, "F1": 0.630, "RMSE": 0.540},
        "EPSP-LLM(ours)": {"Acc.": 0.722, "Prec.": 0.689, "Rec.": 0.682, "F1": 0.685, "RMSE": 0.527}
    },
    "8:2": {
        "LR": {"Acc.": 0.671, "Prec.": 0.621, "Rec.": 0.554, "F1": 0.532, "RMSE": 0.573},
        "KNN": {"Acc.": 0.684, "Prec.": 0.648, "Rec.": 0.573, "F1": 0.559, "RMSE": 0.562},
        "SVM": {"Acc.": 0.633, "Prec.": 0.595, "Rec.": 0.597, "F1": 0.596, "RMSE": 0.605},
        "NB": {"Acc.": 0.679, "Prec.": 0.634, "Rec.": 0.565, "F1": 0.544, "RMSE": 0.548},
        "DT": {"Acc.": 0.687, "Prec.": 0.676, "Rec.": 0.555, "F1": 0.521, "RMSE": 0.562},
        "RF": {"Acc.": 0.664, "Prec.": 0.610, "Rec.": 0.583, "F1": 0.576, "RMSE": 0.572},
        "MLP": {"Acc.": 0.620, "Prec.": 0.536, "Rec.": 0.525, "F1": 0.514, "RMSE": 0.616},
        "GBC": {"Acc.": 0.684, "Prec.": 0.640, "Rec.": 0.590, "F1": 0.588, "RMSE": 0.562},
        "ACO-DT": {"Acc.": 0.716, "Prec.": 0.681, "Rec.": 0.621, "F1": 0.626, "RMSE": 0.534},
        "ANFIS": {"Acc.": 0.701, "Prec.": 0.662, "Rec.": 0.619, "F1": 0.626, "RMSE": 0.546},
        "EPSP-LLM(ours)": {"Acc.": 0.722, "Prec.": 0.689, "Rec.": 0.682, "F1": 0.685, "RMSE": 0.527}
    }, # 数据同上，简略省略
    "5:5": {
        "LR": {"Acc.": 0.674, "Prec.": 0.633, "Rec.": 0.568, "F1": 0.542, "RMSE": 0.569},
        "KNN": {"Acc.": 0.687, "Prec.": 0.644, "Rec.": 0.583, "F1": 0.563, "RMSE": 0.565},
        "SVM": {"Acc.": 0.629, "Prec.": 0.584, "Rec.": 0.586, "F1": 0.588, "RMSE": 0.658},
        "NB": {"Acc.": 0.682, "Prec.": 0.641, "Rec.": 0.571, "F1": 0.551, "RMSE": 0.539},
        "DT": {"Acc.": 0.688, "Prec.": 0.684, "Rec.": 0.570, "F1": 0.529, "RMSE": 0.554},
        "RF": {"Acc.": 0.671, "Prec.": 0.622, "Rec.": 0.591, "F1": 0.583, "RMSE": 0.568},
        "MLP": {"Acc.": 0.628, "Prec.": 0.543, "Rec.": 0.537, "F1": 0.527, "RMSE": 0.602},
        "GBC": {"Acc.": 0.695, "Prec.": 0.655, "Rec.": 0.605, "F1": 0.598, "RMSE": 0.523},
        "ACO-DT": {"Acc.": 0.718, "Prec.": 0.682, "Rec.": 0.619, "F1": 0.625, "RMSE": 0.532},
        "ANFIS": {"Acc.": 0.711, "Prec.": 0.671, "Rec.": 0.627, "F1": 0.632, "RMSE": 0.535},
        "EPSP-LLM(ours)": {"Acc.": 0.722, "Prec.": 0.689, "Rec.": 0.682, "F1": 0.685, "RMSE": 0.527}
    } # 数据同上，简略省略
}

# 转化为 DataFrame
rows = []
for split, methods in data.items():
    for method, metrics in methods.items():
        rows.append({"Split": split, "Method": method, **metrics})

df = pd.DataFrame(rows)

# 定义颜色映射
methods = df['Method'].unique()
# palette = sns.color_palette("Paired", len(methods))
palette = sns.color_palette("bright", len(methods)) 
color_map = dict(zip(methods, palette))

# 指标列表
metrics = ["Acc.", "Prec.", "Rec.", "F1", "RMSE"]

# 定义字体属性
axis_font = {'size': 14, 'weight': 'bold'}
label_font = {'size': 16, 'weight': 'bold'}
title_font = {'size': 18, 'weight': 'bold'}
legend_font = {'size': 12, 'weight': 'bold'}

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
    # 设置标题和轴标签字体
    ax.set_title(f"{metric} Comparison", fontdict=title_font)
    ax.set_xlabel("Split Ratio", fontdict=label_font)
    ax.set_ylabel(metric, fontdict=label_font)

    # 设置x轴和y轴刻度标签字体加粗
    for tick in ax.get_xticklabels():
        tick.set_fontsize(16)  # 设置字体大小
        tick.set_fontweight('bold')  # 设置字体加粗

    for tick in ax.get_yticklabels():
        tick.set_fontsize(16)  # 设置字体大小
        tick.set_fontweight('bold')  # 设置字体加粗

    # 设置y轴的起始值
    metric_values = df[metric].values
    min_y = np.floor(np.min(metric_values) * 10) / 10  # 向下取整
    ax.set_ylim(bottom=min_y)

    # 移除图例
    ax.get_legend().remove()

# 获取图例句柄和标签
handles, labels = axes[0].get_legend_handles_labels()

# 创建主图例，设置字体属性
fig.legend(
    handles, labels, title="", loc="upper center",
    ncol=len(methods), bbox_to_anchor=(0.5, 1),
    prop=legend_font,
    handleheight=1.5,  # 增大颜色块高度
    handlelength=3,  # 增大颜色块长度
    labelspacing=4  # 增加图例项间距
)


# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.94])  # 留出图例空间

# 保存图片
plt.savefig(os.path.join(save_dir, "experiment_2.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "experiment_2.pdf"))  # 保存为PDF
plt.show()