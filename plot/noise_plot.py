import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15
})

# Read the Excel file
df = pd.read_excel('/home/lh/Dowzag_2.0/plot/noise_mrr.xlsx')

# Set the width of each bar and positions of the bars
width = 0.35
x = np.arange(len(df.columns[1:]))  # 从第二列开始，跳过第一列

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 4))

# Create bars
rects1 = ax.bar(x - width/2, df.iloc[0, 1:], width, label='WinGNN')
rects2 = ax.bar(x + width/2, df.iloc[1, 1:], width, label='TMetaNet')

# Customize the plot
ax.set_ylabel('MRR')
ax.set_xticks(x)
ax.set_xticklabels(df.columns[1:])  # 使用从第二列开始的列名作为x轴标签
ax.legend()

# 设置y轴范围
y_min = df.iloc[:, 1:].values.min()  # 直接使用数据的最小值
y_max = df.iloc[:, 1:].values.max() + 0.005
# 设置一个小的边距（这里用5%的数据范围）
margin = (y_max - y_min) * 0.05
ax.set_ylim(y_min - margin, y_max + margin)

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Adjust layout and display
plt.tight_layout()
plt.savefig('/home/lh/Dowzag_2.0/plot/noise_mrr.png')
plt.show()
