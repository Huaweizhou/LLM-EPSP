import numpy as np
import matplotlib.pyplot as plt

# 实验数据
missing_rates = np.arange(10, 70, 10)  # Missing Rate 从10%到60%
template_f1 = [0.675, 0.604, 0.554, 0.511, 0.441, 0.351]  # 模板提示性能
casual_f1 = [0.562, 0.498, 0.462, 0.426, 0.335, 0.222]    # 随意提示性能

# 绘制柱状图
bar_width = 0.35
x = np.arange(len(missing_rates))  # 确保 x 的范围与 missing_rates 一致

plt.figure(figsize=(10, 6))  # 调整图表大小
plt.bar(x - bar_width / 2, template_f1, width=bar_width, color="blue", 
        label=r"$\mathbf{Template\ Prompts}$", alpha=0.8)  # 加粗 label
plt.bar(x + bar_width / 2, casual_f1, width=bar_width, color="red", 
        label=r"$\mathbf{Casual\ Prompts}$", alpha=0.8)  # 加粗 label

# 调整刻度
plt.xticks(x, [f"{rate}%" for rate in missing_rates], fontsize=14, weight="bold")  # 加粗 x 轴刻度
plt.yticks(np.arange(0.2, 0.8, 0.1), fontsize=14, weight="bold")  # 加粗 y 轴刻度

# 图表设置
plt.xlabel("Missing Rate", fontsize=14, weight="bold")  # 加粗字体
plt.ylabel("F1 Score", fontsize=14, weight="bold")  # 加粗字体
plt.title("Performance Comparison: Template vs Casual Prompts", fontsize=16, weight="bold")  # 设置标题字体
plt.legend(fontsize=12, loc="upper right")  # 调整图例位置和字体大小
plt.ylim(0.2, 0.7)  # 限制 y 轴范围
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 调整 x 轴范围和显示的范围
plt.xlim(-0.5, len(missing_rates) - 0.5)  # 确保 x 轴范围完整

# 使用 for 循环对刻度数字加粗
ax = plt.gca()  # 获取当前轴
for tick in ax.get_xticklabels():  # 对 x 轴刻度数字加粗
    tick.set_fontweight("bold")
for tick in ax.get_yticklabels():  # 对 y 轴刻度数字加粗
    tick.set_fontweight("bold")

plt.tight_layout()

# 保存图片
plt.savefig("data_anysics/prompt_template.png", dpi=300)  # 保存更高分辨率图片
plt.savefig("data_anysics/prompt_template.pdf")
plt.show()
