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
        "LR": {"Acc.": 0.711, "Prec.": 0.628, "Rec.": 0.652, "F1": 0.634,"RMSE": 0.561},
        "KNN": {"Acc.": 0.703, "Prec.": 0.593, "Rec.": 0.678, "F1": 0.618, "RMSE": 0.575},
        "SVM": {"Acc.": 0.714, "Prec.": 0.599, "Rec.": 0.667, "F1": 0.625, "RMSE": 0.563},
        "NB": {"Acc.": 0.733, "Prec.": 0.647, "Rec.": 0.671, "F1": 0.659,"RMSE": 0.558},
        "DT": {"Acc.": 0.731, "Prec.": 0.623, "Rec.": 0.658, "F1": 0.624, "RMSE": 0.586},
        "RF": {"Acc.": 0.717, "Prec.": 0.606, "Rec.": 0.669, "F1": 0.619, "RMSE": 0.575},
        "MLP": {"Acc.": 0.689, "Prec.": 0.594, "Rec.": 0.689, "F1": 0.627, "RMSE": 0.598},
        "GBC": {"Acc.": 0.739, "Prec.": 0.651, "Rec.": 0.674, "F1": 0.638, "RMSE": 0.521},
        "ACO-DT": {"Acc.": 0.746, "Prec.": 0.658, "Rec.": 0.696, "F1": 0.651, "RMSE": 0.515},
        "ANFIS": {"Acc.": 0.755, "Prec.": 0.661, "Rec.": 0.712, "F1": 0.645, "RMSE": 0.520},
        "EPSP-LLM(ours)": {"Acc.": 0.759, "Prec.": 0.665, "Rec.": 0.736, "F1": 0.679, "RMSE": 0.490}
        # "ours": {"Acc.": 0.722, "Prec.": 0.689, "Rec.": 0.682, "F1": 0.685, "RMSE": 0.527}
    },
    "8:2": {
        "LR": {"Acc.": 0.715, "Prec.": 0.632, "Rec.": 0.655, "F1": 0.631, "RMSE": 0.566},
        "KNN": {"Acc.": 0.692, "Prec.": 0.588, "Rec.": 0.677, "F1": 0.629, "RMSE": 0.564},
        "SVM": {"Acc.": 0.719, "Prec.": 0.607, "Rec.": 0.658, "F1": 0.616, "RMSE": 0.525},
        "NB": {"Acc.": 0.737, "Prec.": 0.644, "Rec.": 0.674, "F1": 0.654, "RMSE": 0.541},
        "DT": {"Acc.": 0.730, "Prec.": 0.626, "Rec.": 0.652, "F1": 0.621, "RMSE": 0.563},
        "RF": {"Acc.": 0.704, "Prec.": 0.610, "Rec.": 0.666, "F1": 0.627, "RMSE": 0.564},
        "MLP": {"Acc.": 0.697, "Prec.": 0.606, "Rec.": 0.678, "F1": 0.634, "RMSE": 0.596},
        "GBC": {"Acc.": 0.742, "Prec.": 0.662, "Rec.": 0.686, "F1": 0.658, "RMSE": 0.522},
        "ACO-DT": {"Acc.": 0.749, "Prec.": 0.667, "Rec.": 0.702, "F1": 0.659, "RMSE": 0.519},
        "ANFIS": {"Acc.": 0.751, "Prec.": 0.664, "Rec.": 0.708, "F1": 0.656, "RMSE": 0.508},
        "EPSP-LLM(ours)": {"Acc.": 0.759, "Prec.": 0.665, "Rec.": 0.736, "F1": 0.679, "RMSE": 0.490}
    }, 
    "5:5": {
        "LR": {"Acc.": 0.708, "Prec.": 0.633, "Rec.": 0.663, "F1": 0.643, "RMSE": 0.568},
        "KNN": {"Acc.": 0.714, "Prec.": 0.584, "Rec.": 0.683, "F1": 0.613, "RMSE": 0.551},
        "SVM": {"Acc.": 0.719, "Prec.": 0.594, "Rec.": 0.656, "F1": 0.618, "RMSE": 0.538},
        "NB": {"Acc.": 0.735, "Prec.": 0.641, "Rec.": 0.678, "F1": 0.652, "RMSE": 0.533},
        "DT": {"Acc.": 0.738, "Prec.": 0.654, "Rec.": 0.671, "F1": 0.621, "RMSE": 0.558},
        "RF": {"Acc.": 0.703, "Prec.": 0.612, "Rec.": 0.661, "F1": 0.633, "RMSE": 0.567},
        "MLP": {"Acc.": 0.706, "Prec.": 0.613, "Rec.": 0.667, "F1": 0.624, "RMSE": 0.582},
        "GBC": {"Acc.": 0.744, "Prec.": 0.665, "Rec.": 0.695, "F1": 0.668, "RMSE": 0.510},
        "ACO-DT": {"Acc.": 0.743, "Prec.": 0.672, "Rec.": 0.701, "F1": 0.655, "RMSE": 0.524},
        "ANFIS": {"Acc.": 0.752, "Prec.": 0.661, "Rec.": 0.713, "F1": 0.652, "RMSE": 0.501},
       "EPSP-LLM(ours)": {"Acc.": 0.759, "Prec.": 0.665, "Rec.": 0.736, "F1": 0.679, "RMSE": 0.490}
    } 
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
        tick.set_fontsize(16)  
        tick.set_fontweight('bold')  

    for tick in ax.get_yticklabels():
        tick.set_fontsize(16)  
        tick.set_fontweight('bold')  

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
plt.savefig(os.path.join(save_dir, "experiment_1.png"), dpi=300)
plt.savefig(os.path.join(save_dir, "experiment_1.pdf"))  # 保存为PDF
plt.show()