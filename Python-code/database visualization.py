import pandas as pd
import matplotlib.pyplot as plt
# CSV 文件路径
file_path = 'Flight Attendant Report.csv'
# 读取 CSV 文件
df = pd.read_csv(file_path)

df['Attendant Name'].value_counts().plot(kind='bar')
plt.title('Number of Flights per Attendant')
plt.xlabel('Attendant Name')
plt.ylabel('Number of Flights')
plt.show()

# 绘制饼图 - 各乘务员的总飞行里程占比
df.groupby('Attendant Name')['Mileage'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Total Mileage by Attendant')
plt.ylabel('')  # 隐藏 Y 轴标签
plt.show()

# 绘制直方图 - 航程的分布
df['Mileage'].plot(kind='hist', bins=10)
plt.title('Distribution of Flight Mileage')
plt.xlabel('Mileage')
plt.ylabel('Frequency')
plt.show()

# 绘制折线图 - 随时间变化的里程
df_sorted = df.sort_values('Flight Date')
plt.plot(df_sorted['Flight Date'], df_sorted['Mileage'])
plt.title('Mileage Over Time')
plt.xlabel('Flight Date')
plt.ylabel('Mileage')
plt.show()