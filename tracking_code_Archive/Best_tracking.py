#The best

import cv2
import numpy as np

class PupilDetector:
    def __init__(self):
        # 加载眼睛检测器
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # 优化参数
        self.eye_params = {
            'scaleFactor': 1.1,    # 检测尺度缩放因子
            'minNeighbors': 8,     # 更高值减少误检
            'minSize': (40, 40)    # 最小检测尺寸
        }
        
        # 瞳孔检测参数
        self.pupil_params = {
            'threshold': 30,       # 二值化阈值
            'min_area': 30,       # 最小瞳孔面积
            'max_area': 1000       # 最大瞳孔面积
        }
    
    def detect_pupil(self, eye_region):
        """在眼睛区域内检测单个瞳孔"""
        # 灰度化 + 高斯模糊
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        
        # 自适应阈值处理
        _, thresh = cv2.threshold(gray_eye, self.pupil_params['threshold'], 
                                255, cv2.THRESH_BINARY_INV)
        
        # 形态学处理去噪
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # 只保留面积最大的一个轮廓
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        # 面积过滤
        if not (self.pupil_params['min_area'] < area < self.pupil_params['max_area']):
            return None
        
        # 获取最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        return (int(cx), int(cy)), int(radius)

    def process_frame(self, frame):
        """处理单帧图像"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 增强对比度
        
        # 检测眼睛
        eyes = self.eye_cascade.detectMultiScale(gray, **self.eye_params)
        
        for (x, y, w, h) in eyes:
            # 绘制眼睛矩形框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 提取眼睛区域
            eye_region = frame[y:y+h, x:x+w]
            
            # 检测瞳孔
            pupil_data = self.detect_pupil(eye_region)
            if pupil_data:
                (px, py), radius = pupil_data
                
                # 绘制瞳孔(红色圆圈+中心点)
                cv2.circle(eye_region, (px, py), radius, (0, 0, 255), 2)
                cv2.circle(eye_region, (px, py), 2, (255, 0, 0), -1)
                
                # 显示坐标信息
                global_x, global_y = x + px, y + py
                cv2.putText(frame, f"Pupil: ({global_x}, {global_y})", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 1)
                
                print(f"Detected Pupil at: ({global_x}, {global_y}), Radius: {radius}px")
        
        return frame

def main():
    detector = PupilDetector()
    cap = cv2.VideoCapture(2)  # 打开默认摄像头
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧并显示
        processed_frame = detector.process_frame(frame)
        cv2.imshow('Real-time Pupil Detection', processed_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()