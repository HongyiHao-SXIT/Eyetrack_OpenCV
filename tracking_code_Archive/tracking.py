import cv2
import numpy as np
import math
from collections import deque

# 初始化计数器和参数
miss_num = 0
close_my_eye_num = 0
position_change_threshold = 25  # 瞳孔位置变化阈值（像素）

# PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.max_output = max_output  # 最大输出限制
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)

    def update(self, setpoint, current_value):
        error = setpoint - current_value

        # 比例项
        p_term = self.kp * error

        # 积分项 (带抗饱和)
        self.integral += error
        i_term = self.ki * self.integral

        # 微分项
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error.copy()

        # 计算总输出
        output = p_term + i_term + d_term

        # 限制输出范围
        output = np.clip(output, -self.max_output, self.max_output)

        return output


# 插值函数
def interpolate_position(current_pos, target_pos, factor=0.2):
    """在当前位置和目标位置之间插值"""
    return current_pos * (1 - factor) + target_pos * factor


# 初始化PID控制器
pid = PIDController(kp=0.5, ki=0.01, kd=0.1, max_output=20)

# 加载Haar级联分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开输入流，使用 USB 摄像头
cap = cv2.VideoCapture('img/eyes_data.mp4')

# 模拟显示画面设置 (11.0cm x 9.0cm)
display_width, display_height = 440, 360  # 假设1cm=40像素

# 初始化当前输出的坐标，使用浮点数提高精度
current_output_pos = np.array([display_width // 2, display_height // 2], dtype=np.float32)

# 创建显示窗口
cv2.namedWindow('Eye Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Eye Tracking', 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 创建用于显示的副本
    display_frame = frame.copy()

    # 裁剪和灰度处理
    height, width = frame.shape[:2]
    cropped_frame = frame[0:height // 2, 0:width // 2]
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # 眼睛检测
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))

    current_pupil_positions = []
    pupil_detected = False

    for (x, y, w, h) in eyes:
        # 在原始帧上绘制眼睛矩形
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eye_region = gray[y:y + h, x:x + w]

        # 瞳孔检测，使用自适应阈值
        thresh = cv2.adaptiveThreshold(eye_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 30]

        if valid_contours:
            pupil_detected = True
            largest_contour = max(valid_contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            global_cx, global_cy = int(x + cx), int(y + cy)
            current_pupil_positions.append((global_cx, global_cy))

            # 在原始帧上绘制瞳孔位置
            cv2.circle(display_frame, (global_cx, global_cy), int(radius), (0, 0, 255), 2)

    # 检查瞳孔状态
    if not pupil_detected:
        print("close_my_eye")
        close_my_eye_num += 1
        # 当无法检测到瞳孔时，让输出的坐标慢慢回中
        center_pos = np.array([display_width // 2, display_height // 2], dtype=np.float32)
        current_output_pos = interpolate_position(current_output_pos, center_pos, factor=0.05)
    elif current_pupil_positions:
        pupil_x, pupil_y = current_pupil_positions[0]

        # 计算目标位置
        target_x = np.clip(int(pupil_x * display_width / (width // 2)), 25, display_width - 25)
        target_y = np.clip(int(pupil_y * display_height / (height // 2)), 25, display_height - 25)
        target_pos = np.array([target_x, target_y], dtype=np.float32)

        # 使用PID控制器更新位置
        pid_output = pid.update(target_pos, current_output_pos)
        current_output_pos += pid_output

    # 在原始帧上绘制输出位置（映射回原始图像坐标）
    output_x = int(current_output_pos[0] * (width // 2) / display_width)
    output_y = int(current_output_pos[1] * (height // 2) / display_height)
    cv2.circle(display_frame, (output_x, output_y), 10, (255, 0, 0), -1)

    # 显示状态文本
    status_text = f"Tracking: {'ON' if pupil_detected else 'OFF'}"
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Position: ({int(current_output_pos[0])}, {int(current_output_pos[1])})",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 输出当前坐标
    print(f"Output coordinates: x={int(current_output_pos[0])}, y={int(current_output_pos[1])}")

    # 显示处理后的帧
    cv2.imshow('Eye Tracking', display_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 输出结果
print("\nThis is the conclusion of the test:")
print(f"Your eye was miss: {miss_num} times")
print(f"Your eye was close: {close_my_eye_num} times")

cap.release()
cv2.destroyAllWindows()
    