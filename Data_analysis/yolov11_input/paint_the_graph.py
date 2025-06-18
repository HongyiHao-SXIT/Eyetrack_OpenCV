import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 原始数据（确保列名与访问时一致）
data = {
    'Metric': ['Total frames processed', 'Overall average confidence'],
    'Stream': [173, 0.6108],
    'Video': [173, 0.4127]
}

# 转换为DataFrame（确保列名正确）
df = pd.DataFrame(data)

# 打印DataFrame确认结构
print("数据预览：")
print(df)

# 设置图表风格
try:
    plt.style.use('seaborn-v0_8')  # 现代Matplotlib中的Seaborn风格
except:
    plt.style.use('ggplot')  # 回退风格

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Performance Comparison: Stream vs Video', fontsize=14, y=1.05)

# 第一幅图：帧数对比
try:
    ax1.bar(['Stream', 'Video'], [df.loc[0, 'Stream'], df.loc[0, 'Video']], 
           color=['#1f77b4', '#ff7f0e'])
    ax1.set_title('Frame Processing Comparison')
    ax1.set_ylabel('Frames')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, val in enumerate([df.loc[0, 'Stream'], df.loc[0, 'Video']]):
        ax1.text(i, val/2, f'{val:.0f}', ha='center', va='center', color='white', fontweight='bold')
except KeyError as e:
    print(f"错误：缺少必要的列 {e}")
    print(f"可用列：{df.columns.tolist()}")

# 第二幅图：置信度对比
try:
    ax2.bar(['Stream', 'Video'], [df.loc[1, 'Stream'], df.loc[1, 'Video']], 
           color=['#1f77b4', '#ff7f0e'])
    ax2.set_title('Confidence Level Comparison')
    ax2.set_ylabel('Confidence Score')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, val in enumerate([df.loc[1, 'Stream'], df.loc[1, 'Video']]):
        ax2.text(i, val, f'{val:.4f}', ha='center', va='bottom')
except KeyError as e:
    print(f"错误：缺少必要的列 {e}")
    print(f"可用列：{df.columns.tolist()}")

# 调整布局
plt.tight_layout()

# 保存图表
output_path = 'performance_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图表已保存至：{output_path}")
plt.show()