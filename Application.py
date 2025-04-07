import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt


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
        
        return frame


class PupilDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pupil_Detection_Application (QT6)")
        self.setGeometry(100, 100, 800, 600)

        # 初始化瞳孔检测器
        self.detector = PupilDetector()

        self.main_widget = QWidget()
        self.layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.exit_btn = QPushButton("Exit")

        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_btn.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 8px;"
        )
        self.exit_btn.setStyleSheet(
            "background-color: #607D8B; color: white; font-weight: bold; padding: 8px;"
        )

        self.stop_btn.setEnabled(False)

        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)
        self.layout.addWidget(self.exit_btn)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        self.cap = None
        self.timer = QTimer()
        self.is_detecting = False

        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.exit_btn.clicked.connect(self.close)

    def start_detection(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            print("Error: Unable to open camera")
            return

        self.is_detecting = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def stop_detection(self):
        self.is_detecting = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        frame = cv2.flip(frame, 1)
        
        # 使用PupilDetector处理帧
        processed_frame = self.detector.process_frame(frame)

        # 转换为Qt图像格式并显示
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PupilDetectionApp()
    window.show()
    sys.exit(app.exec())