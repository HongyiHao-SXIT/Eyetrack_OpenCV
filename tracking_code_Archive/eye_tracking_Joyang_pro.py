import cv2
import numpy as np

# 加载Haar级联分类器
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 打开摄像头
cap = cv2.VideoCapture('img/eyes_data.mp4')

while True:
    # 读取摄像头数据
    ret, frame = cap.read()

    if not ret:
        break

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 眼睛检测
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        # 绘制矩形框标出眼睛
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取眼睛区域
        eye_region = gray[y:y + h, x:x + w]
        eye_frame = frame[y:y + h, x:x + w]

        # 使用更小的阈值化处理来找到瞳孔
        _, thresh = cv2.threshold(eye_region, 20, 255, cv2.THRESH_BINARY_INV)  # 阈值

        # 查找轮廓（瞳孔）
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 只处理面积大于50的轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
        
        if valid_contours:
            # 找到面积最大的轮廓
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # 获取最大轮廓的边界框
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            
            # 绘制圆圈标记瞳孔
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            
            # 在瞳孔中心画点
            cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            
            # 输出瞳孔坐标 (cx, cy)
            print(f"Pupil coordinates: x={int(cx)}, y={int(cy)}")

    # 显示结果
    cv2.imshow('Pupil Detection', frame)

    # 按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()