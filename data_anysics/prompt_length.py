import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline

# 指定横轴和对应的 F1-score 值
f1_scores = np.array([0.47, 0.49, 0.48, 0.49, 0.50, 0.51, 0.52, 0.54, 0.53, 0.55,
                      0.54, 0.55, 0.57, 0.58, 0.56, 0.59, 0.60, 0.61, 0.62, 0.64,
                      0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.681, 0.67, 0.65, 0.66,
                      0.635, 0.628, 0.611, 0.620, 0.615, 0.609, 0.591, 0.583, 0.5774, 0.589])  # F1-score 值数组

# 横轴数据点与 F1-score 数组的长度对应
number_of_words = np.linspace(100, 500, len(f1_scores))  # 根据 F1-score 数组长度调整横轴数据点

# 创建颜色映射
colors = f1_scores

# 创建图表
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    x=number_of_words, y=f1_scores, hue=colors, palette="Reds", size=colors, sizes=(20, 200), legend=None
)

# 使用 UnivariateSpline 创建拟合曲线
spline = UnivariateSpline(number_of_words, f1_scores, s=0.5)  # s 是平滑参数
x_smooth = np.linspace(100, 500, 500)  # 平滑曲线的横轴范围
y_smooth = spline(x_smooth)  # 拟合曲线的纵轴数据
plt.plot(x_smooth, y_smooth, color='darkblue', linestyle='-', linewidth=2, label="Fitted Curve")  # 实线并加深颜色

# 添加颜色条（右侧的图例）
norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())  # 归一化颜色范围
sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)  # 使用红色渐变
sm.set_array([])  # 设置颜色条的数组为空
cbar = plt.colorbar(sm, ax=plt.gca())  # 添加颜色条
# cbar.set_label("F1-Score")  # 设置颜色条标签

# 使用 for 循环加粗颜色条的刻度标签
for tick in cbar.ax.get_yticklabels():
    tick.set_fontsize(12)  # 设置字体大小
    tick.set_color('black')  # 设置字体颜色
    tick.set_fontweight('bold')  # 设置字体加粗
    tick.set_rotation(0)  # 可选：旋转角度

# 图表美化
plt.title("Performance of the Model with Various Numbers of Words in Prompts", fontsize=16, fontweight='bold')
plt.xlabel("Number of Words in a Prompt", fontsize=14, fontweight='bold')
plt.ylabel("F1-Score", fontsize=14, fontweight='bold')
plt.xticks(np.arange(100, 501, 50), fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.ylim(0.38, 0.72)
plt.grid(True, linestyle="--", alpha=0.6)

# 保存和展示图表
plt.tight_layout()
plt.savefig("data_anysics/prompt_length.png")
plt.savefig("data_anysics/prompt_length.pdf")
plt.show()
