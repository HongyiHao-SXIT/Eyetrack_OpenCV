import csv
from datetime import datetime

# 你的处理结果数据
results = {
    "Total frames processed": 173,
    "Overall average confidence": 0.4127,
    "Exit status": "Resources released, program exited cleanly",
    "Processing date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# CSV文件路径
csv_file = "processing_results.csv"

# 写入CSV文件
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # 写入标题行
    writer.writerow(["Metric", "Value"])
    
    # 写入数据行
    for key, value in results.items():
        writer.writerow([key, value])

print(f"Results successfully saved to {csv_file}")