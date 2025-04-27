import cv2
import numpy as np
import serial
import time
import math

# 串口设置（根据实际情况修改端口号）
ser = serial.Serial('COM9', 9600)  # 根据实际端口号修改

# 打开USB摄像头
cap = cv2.VideoCapture(1)

# 设置摄像头分辨率（根据需要调整）
cap.set(3, 640)  # 宽
cap.set(4, 480)  # 高

# 颜色范围（HSV色彩空间），这里是检测白色（绿色test）
lower_white = np.array([40, 100, 100])  # 低范围
upper_white = np.array([80, 255, 255])  # 高范围

# 舵机控制参数
last_x_offset = 0
last_y_offset = 0
max_angle_change = 15  # 每次舵机最大角度变化
angle_update_delay = 0.02  # 每次调整后等待0.05秒
acceleration_factor = 0.1  # 加速度因子，控制平滑加速/减速

# 视角中心点
frame_center = (320, 240)  # 摄像头分辨率设定为640x480

# PID 控制器参数
Kp = 0.01  # 比例常数
Ki = 0.00039  # 积分常数
Kd = 0.05  # 微分常数

# PID 控制器状态
previous_error_x = 0
integral_x = 0
previous_error_y = 0
integral_y = 0

# 最大速度
max_speed = 20  # 最大速度限制

# 死区阈值：偏差小于该值时不进行任何调整
dead_zone = 0  # 死区阈值

# 平滑参数
smooth_factor = 0.09  # 控制平滑度，数值越小，平滑效果越强

# 防抖
motion_smooth_factor = 0.8  # 控制防抖的平滑系数
prev_center = (0, 0)  # 上一帧目标位置

# 主循环
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 高斯模糊
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # 创建一个白色物体掩码
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 对掩码进行形态学处理，去除噪点
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 过滤小轮廓，只关注足够大的物体
        contours = [c for c in contours if cv2.contourArea(c) > 1000]  # 可以根据需要调整最小面积

        if contours:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)

            # 获取最小外接矩形
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)

            # 计算矩形的中心点
            center = (int(rect[0][0]), int(rect[0][1]))

            # 在图像上画出矩形和中心点
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # 防抖处理：计算当前帧中心与上一帧中心的偏差
            if prev_center != (0, 0):  # 第一次处理时跳过防抖
                x_diff = abs(center[0] - prev_center[0])
                y_diff = abs(center[1] - prev_center[1])

                # 如果偏差过小，认为是抖动，忽略本次调整
                if x_diff < 5 and y_diff < 5:
                    center = prev_center  # 使用上一帧的位置，避免小幅度抖动

            # 更新上一帧的目标位置
            prev_center = center

            # 计算目标相对于画面中心的偏差
            x_offset = -(center[0] - frame_center[0])  # 反转 x 轴偏移
            y_offset = center[1] - frame_center[1]

            # 死区控制：如果偏差小于死区范围，不做任何调整
            if abs(x_offset) < dead_zone:
                x_offset = 0
            if abs(y_offset) < dead_zone:
                y_offset = 0

            # 计算偏移量的距离（欧几里得距离）
            distance = math.sqrt(x_offset**2 + y_offset**2)

            # 动态调整舵机速度：根据偏移距离控制速度
            speed_factor = min(1 + (distance / frame_center[0]), max_speed)  # 控制速度，但不超过最大值

            # PID 控制：x 轴
            error_x = x_offset
            integral_x += error_x
            derivative_x = error_x - previous_error_x
            previous_error_x = error_x
            pid_x = Kp * error_x + Ki * integral_x + Kd * derivative_x

            # PID 控制：y 轴
            error_y = y_offset
            integral_y += error_y
            derivative_y = error_y - previous_error_y
            previous_error_y = error_y
            pid_y = Kp * error_y + Ki * integral_y + Kd * derivative_y

            # 将 PID 控制的值进行平滑
            smoothed_x_offset = pid_x * speed_factor
            smoothed_y_offset = pid_y * speed_factor

            # 平滑控制：加权平均
            smoothed_x_offset = smooth_factor * smoothed_x_offset + (1 - smooth_factor) * last_x_offset
            smoothed_y_offset = smooth_factor * smoothed_y_offset + (1 - smooth_factor) * last_y_offset

            # 限制舵机的最大角度变化
            smoothed_x_offset = min(max(smoothed_x_offset, -max_angle_change), max_angle_change)
            smoothed_y_offset = min(max(smoothed_y_offset, -max_angle_change), max_angle_change)

            # 保存上次的偏移量
            last_x_offset = smoothed_x_offset
            last_y_offset = smoothed_y_offset

            # 向Arduino发送平滑后的偏移信息
            ser.write(f"{int(smoothed_x_offset)},{int(smoothed_y_offset)}\n".encode())

    # 显示图像
    cv2.imshow('Frame', frame)

    # 每次调整后暂停，控制更新频率
    time.sleep(angle_update_delay)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和串口资源
cap.release()
ser.close()
cv2.destroyAllWindows()
