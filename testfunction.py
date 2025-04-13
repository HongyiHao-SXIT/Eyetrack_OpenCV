import cv2
import numpy as np
import math

miss_num = 0
close_my_eye_num = 0

# 加载Haar级联分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开输入流
cap = cv2.VideoCapture('img/eyes_data.mp4')

# 初始化前一帧的瞳孔位置
prev_pupil_positions = []
position_change_threshold = 25  # 瞳孔位置变化阈值（像素）

while True:
    # 读取视频帧
    ret, frame = cap.read()

    if not ret:
        break

    # 裁剪画面为原来的一半（高度和宽度各减半）
    height, width = frame.shape[:2]
    cropped_frame = frame[0:height//2, 0:width//2]
    
    # 转为灰度图（在裁剪后的画面上处理）
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # 眼睛检测
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    current_pupil_positions = []
    pupil_detected = False

    for (x, y, w, h) in eyes:
        # 绘制矩形框标出眼睛
        cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取眼睛区域
        eye_region = gray[y:y + h, x:x + w]
        eye_frame = cropped_frame[y:y + h, x:x + w]

        # 使用更小的阈值化处理来找到瞳孔
        _, thresh = cv2.threshold(eye_region, 20, 255, cv2.THRESH_BINARY_INV)

        # 查找轮廓（瞳孔）
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 只处理面积大于25的轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
        
        if valid_contours:
            pupil_detected = True
            # 找到面积最大的轮廓
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # 获取最大轮廓的边界框
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            global_cx, global_cy = int(x + cx), int(y + cy)  # 转换为全局坐标
            
            current_pupil_positions.append((global_cx, global_cy))
            
            # 绘制圆圈标记瞳孔
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            
            # 在瞳孔中心画点
            cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            
            # 输出瞳孔坐标 (cx, cy)
            print(f"Pupil coordinates: x={global_cx}, y={global_cy}")

    # 检查瞳孔状态
    if not pupil_detected:
        print("close_my_eye")
        close_my_eye_num += 1
    elif prev_pupil_positions and current_pupil_positions:
        # 计算瞳孔位置变化
        position_changes = []
        for prev_pos, curr_pos in zip(prev_pupil_positions, current_pupil_positions):
            distance = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            position_changes.append(distance)
        
        # 如果有任何一个瞳孔移动距离超过阈值
        if any(change > position_change_threshold for change in position_changes):
            print("miss")
            miss_num += 1
    
    # 更新前一帧的瞳孔位置
    prev_pupil_positions = current_pupil_positions

    # 显示裁剪后的结果
    cv2.imshow('Pupil Detection (Cropped)', cropped_frame)

    # 按键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 输出最终结果
print("This is the conclusion of the test:")
print(f"Your eye was miss: {miss_num} times")
print(f"Your eye was close: {close_my_eye_num} times")

# 释放资源
cap.release()
cv2.destroyAllWindows()