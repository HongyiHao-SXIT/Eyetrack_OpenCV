import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    
    # 镜像翻转（使画面更自然）
    frame = cv2.flip(frame, 1)
    
    # 定义ROI区域
    roi = frame[100:500, 157:800]  
    
    # 图像处理流程
    rows, cols, _ = roi.shape 
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 8, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # 绘制检测结果
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break

    # 显示结果
    cv2.imshow("Pupil Detection", roi)
    cv2.imshow("Threshold", threshold)
    
    # 退出逻辑（按q键退出）
    if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms延迟，同时检测按键
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()