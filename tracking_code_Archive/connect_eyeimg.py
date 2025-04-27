import cv2
import numpy as np
import math
from collections import deque

# 初始化计数器和参数
miss_num = 0
close_my_eye_num = 0
position_change_threshold = 30  # 瞳孔位置变化阈值（像素）

# 平滑处理参数
SMOOTHING_WINDOW_SIZE = 5  # 移动平均窗口大小
POSITION_HISTORY = deque(maxlen=SMOOTHING_WINDOW_SIZE)  # 存储历史位置用于平滑

# 加载Haar级联分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开输入流
cap = cv2.VideoCapture(2)

# 模拟显示画面设置 (11.0cm x 9.0cm)
display_width, display_height = 440, 360  # 假设1cm=40像素
display_img = np.zeros((display_height, display_width, 3), dtype=np.uint8)

# 读取本地图片
image_path = 'img/eye.png'  # 请将此路径替换为实际的图片路径
controlled_img = cv2.imread(image_path)
if controlled_img is None:
    print("无法读取图片，请检查图片路径。")
    exit()

# 调整图片大小
controlled_img = cv2.resize(controlled_img, (50, 50))

controlled_pos = np.array([display_width//2, display_height//2], dtype=np.float32)  # 使用浮点数提高精度

# 初始化前一帧的瞳孔位置
prev_pupil_positions = []

def apply_smoothing(new_pos):
    """应用平滑滤波到新位置"""
    POSITION_HISTORY.append(new_pos)
    
    # 如果历史数据不足，直接返回新位置
    if len(POSITION_HISTORY) < 2:
        return new_pos
    
    # 计算加权移动平均 (最近的位置权重更高)
    weights = np.linspace(0.1, 1.0, len(POSITION_HISTORY))
    weights /= weights.sum()  # 归一化
    
    smoothed_pos = np.zeros(2)
    for i, pos in enumerate(POSITION_HISTORY):
        smoothed_pos += pos * weights[i]
    
    return smoothed_pos

def interpolate_position(current_pos, target_pos, factor=0.2):
    """在当前位置和目标位置之间插值"""
    return current_pos * (1 - factor) + target_pos * factor

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 裁剪和灰度处理
    height, width = frame.shape[:2]
    cropped_frame = frame[0:height//2, 0:width//2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 眼睛检测
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    current_pupil_positions = []
    pupil_detected = False

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        eye_region = gray[y:y + h, x:x + w]
        eye_frame = frame[y:y + h, x:x + w]

        # 瞳孔检测
        _, thresh = cv2.threshold(eye_region, 20, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
        
        if valid_contours:
            pupil_detected = True
            largest_contour = max(valid_contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            global_cx, global_cy = int(x + cx), int(y + cy)
            current_pupil_positions.append((global_cx, global_cy))
            
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            print(f"Pupil coordinates: x={global_cx}, y={global_cy}")

    # 检查瞳孔状态
    if not pupil_detected:
        print("close_my_eye")
        close_my_eye_num += 1
    elif prev_pupil_positions and current_pupil_positions:
        position_changes = [
            math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
            for prev, curr in zip(prev_pupil_positions, current_pupil_positions)
        ]
        if any(change > position_change_threshold for change in position_changes):
            print("miss")
            miss_num += 1
    
    prev_pupil_positions = current_pupil_positions

    # 更新模拟显示画面
    display_img[:] = (240, 240, 240)  # 灰色背景
    
    if current_pupil_positions:
        pupil_x, pupil_y = current_pupil_positions[0]
        
        # 计算目标位置
        target_x = np.clip(int(pupil_x * display_width / (width//2)), 25, display_width-25)
        target_y = np.clip(int(pupil_y * display_height / (height//2)), 25, display_height-25)
        target_pos = np.array([target_x, target_y], dtype=np.float32)
        
        # 应用平滑处理
        smoothed_target = apply_smoothing(target_pos)
        controlled_pos = interpolate_position(controlled_pos, smoothed_target)
        
        # 转换为整数坐标
        draw_pos = controlled_pos.astype(int)
        x, y = draw_pos[0], draw_pos[1]
        
        # 绘制控制对象
        display_img[y-25:y+25, x-25:x+25] = controlled_img
    
    # 绘制边框和标注
    cv2.rectangle(display_img, (0, 0), (display_width-1, display_height-1), (0, 0, 0), 2)
    cv2.putText(display_img, "11.0cm x 9.0cm", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 显示结果
    cv2.imshow('Pupil Detection (Cropped)', frame)
    cv2.imshow('Eye-Controlled Display (11.0cm x 9.0cm)', display_img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 输出结果
print("\nThis is the conclusion of the test:")
print(f"Your eye was miss: {miss_num} times")
print(f"Your eye was close: {close_my_eye_num} times")

cap.release()
cv2.destroyAllWindows()