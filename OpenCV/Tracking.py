import cv2
import numpy as np

# 加载Haar级联分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计算裁剪后的宽度和高度
cropped_width = width // 2
cropped_height = height // 2

# 定义视频编码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (cropped_width, cropped_height))

while True:
    # 读取摄像头数据
    ret, frame = cap.read()

    if not ret:
        break

    # 裁剪视频为左上角四分之一
    cropped_frame = frame[:cropped_height, :cropped_width]

    # 转为灰度图
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # 眼睛检测
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        # 绘制矩形框标出眼睛
        cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取眼睛区域
        eye_region = gray[y:y + h, x:x + w]
        eye_frame = cropped_frame[y:y + h, x:x + w]

        # 使用更小的阈值化处理来找到瞳孔
        _, thresh = cv2.threshold(eye_region, 60, 255, cv2.THRESH_BINARY_INV)  # 阈值

        # 查找轮廓（瞳孔）
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None

        # 遍历所有轮廓，找到面积最大的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50 and area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            # 获取轮廓的边界框
            (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)  # 绘制圆圈

            # 输出单个瞳孔坐标 (cx, cy)
            print(f"Pupil coordinates: x={int(cx)}, y={int(cy)}")

    # 显示结果
    cv2.imshow('Pupil Detection', cropped_frame)

    # 写入输出视频
    out.write(cropped_frame)

    # 按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()