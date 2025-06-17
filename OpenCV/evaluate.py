import cv2
import numpy as np
import os
import glob

# 配置数据集路径（你需要根据你的实际路径调整）
image_folder = r'C:\Users\Lanyi\Desktop\Project\Eyetrack_Fursuit\Dataset\images\test'
label_folder = r'C:\Users\Lanyi\Desktop\Project\Eyetrack_Fursuit\Dataset\labels\test'

# 初始化 Haar Cascade 分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

# 误差统计
errors = []
detected_count = 0
missed_count = 0

# 获取所有图片路径
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

for image_path in image_paths:
    image_name = os.path.basename(image_path).replace('.jpg', '')
    label_path = os.path.join(label_folder, image_name + '.txt')

    # 读取图像和标签
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    if not os.path.exists(label_path):
        print(f"Label not found for {image_name}")
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            print(f"No labels in {label_path}")
            continue
        # 默认只读取第一个标签（单目标检测）
        label = lines[0].strip().split()

    if len(label) != 5:
        print(f"Label format error for {image_name}")
        continue

    class_id, x_center_norm, y_center_norm, w_norm, h_norm = map(float, label)
    gt_x = x_center_norm * width
    gt_y = y_center_norm * height

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) == 0:
        print(f"Missed detection in {image_name}")
        missed_count += 1
        continue

    # 只取第一个检测到的眼睛
    (x, y, w, h) = eyes[0]
    eye_region = gray[y:y + h, x:x + w]

    _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

        # 转换为全图坐标
        pupil_x = x + cx
        pupil_y = y + cy

        # 计算欧氏误差
        error = np.sqrt((pupil_x - gt_x) ** 2 + (pupil_y - gt_y) ** 2)
        errors.append(error)
        detected_count += 1

        print(f"{image_name}: Detection Error = {error:.2f} pixels")
    else:
        print(f"No valid pupil contour in {image_name}")
        missed_count += 1

# 统计结果
if detected_count > 0:
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    std_error = np.std(errors)

    print("\n=== Detection Accuracy Summary ===")
    print(f"Total Images: {len(image_paths)}")
    print(f"Detected: {detected_count}")
    print(f"Missed: {missed_count}")
    print(f"Mean Error: {mean_error:.2f} pixels")
    print(f"Max Error: {max_error:.2f} pixels")
    print(f"Standard Deviation: {std_error:.2f} pixels")
else:
    print("No detections were made.")
