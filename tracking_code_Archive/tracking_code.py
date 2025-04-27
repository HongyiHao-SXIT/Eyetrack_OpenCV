import cv2
import numpy as np
import math
from collections import deque

# 初始化计数器和参数
miss_num = 0
close_my_eye_num = 0
position_change_threshold = 25  # 瞳孔位置变化阈值（像素）

# 平滑处理参数
SMOOTHING_WINDOW_SIZE = 7  # 移动平均窗口大小
POSITION_HISTORY = deque(maxlen=SMOOTHING_WINDOW_SIZE)  # 存储历史位置用于平滑

# 加载Haar级联分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开输入流，使用 USB 摄像头
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 裁剪和灰度处理
    height, width = frame.shape[:2]
    cropped_frame = frame[0:height//2, 0:width//2]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # 眼睛检测
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))
    
    current_pupil_positions = []
    pupil_detected = False

    for (x, y, w, h) in eyes:
        eye_region = gray[y:y + h, x:x + w]
        eye_frame = cropped_frame[y:y + h, x:x + w]

        # 瞳孔检测
        _, thresh = cv2.threshold(eye_region, 15, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 30]
        
        if valid_contours:
            pupil_detected = True
            largest_contour = max(valid_contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            global_cx, global_cy = int(x + cx), int(y + cy)
            current_pupil_positions.append((global_cx, global_cy))
            
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            print(f"Pupil coordinates: x={global_cx}, y={global_cy}")

            # 将坐标写入文件
            with open('pupil_coordinates.txt', 'w') as file:
                file.write(f"{global_cx},{global_cy}")

    # 检查瞳孔状态
    if not pupil_detected:
        print("close_my_eye")
        close_my_eye_num += 1
        # 当无法检测到瞳孔时，写入默认坐标（可根据需求调整）
        with open('pupil_coordinates.txt', 'w') as file:
            file.write(f"{width//4},{height//4}")

    # 显示结果
    cv2.imshow('Pupil Detection (Cropped)', cropped_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 输出结果
print("\nThis is the conclusion of the test:")
print(f"Your eye was miss: {miss_num} times")
print(f"Your eye was close: {close_my_eye_num} times")

cap.release()
cv2.destroyAllWindows()