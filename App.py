import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import QTimer, Qt, QRectF, QPointF

class EyeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 200)
        self.pupil_pos = QPointF(150, 100)  # 初始瞳孔位置(居中)
        self.pupil_radius = 30

    def set_pupil_position(self, x, y):
        self.pupil_pos = QPointF(x, y)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制眼白
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QRectF(50, 50, 200, 100))
        
        # 绘制虹膜(蓝色)
        painter.setBrush(QColor(100, 150, 255))
        iris_radius = 40
        painter.drawEllipse(self.pupil_pos, iris_radius, iris_radius)
        
        # 绘制瞳孔(黑色)
        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(self.pupil_pos, self.pupil_radius, self.pupil_radius)

class PupilDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pupil Tracking System")
        self.setGeometry(100, 100, 1000, 600)

        # 初始化瞳孔检测器
        self.detector = PupilDetector()
        
        # 主界面布局
        main_widget = QWidget()
        layout = QHBoxLayout()
        
        # 摄像头视图
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)
        
        # 虚拟眼睛视图
        self.eye_widget = EyeWidget()
        
        # 控制按钮
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Stop Tracking")
        self.exit_btn = QPushButton("Exit")
        
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.stop_btn.setStyleSheet("background-color: #F44336; color: white; padding: 10px;")
        self.exit_btn.setStyleSheet("background-color: #607D8B; color: white; padding: 10px;")
        
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.exit_btn)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        
        # 添加部件到主布局
        layout.addWidget(self.camera_view)
        layout.addWidget(self.eye_widget)
        layout.addWidget(control_panel)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
        # 初始化摄像头和定时器
        self.cap = None
        self.timer = QTimer()
        self.is_tracking = False
        
        # 连接信号槽
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.exit_btn.clicked.connect(self.close)

    def start_tracking(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera not accessible")
            return
            
        self.is_tracking = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)  # 30ms刷新间隔

    def stop_tracking(self):
        self.is_tracking = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.camera_view.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_tracking()
            return
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # 使用改进的瞳孔检测算法
        eyes = self.detector.eye_cascade.detectMultiScale(gray, **self.detector.eye_params)
        
        pupil_positions = []
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            eye_region = frame[y:y+h, x:x+w]
            
            # 检测瞳孔
            pupil_data = self.detector.detect_pupil(eye_region)
            if pupil_data:
                (px, py), radius = pupil_data
                cv2.circle(eye_region, (px, py), radius, (0, 0, 255), 2)
                cv2.circle(eye_region, (px, py), 2, (255, 0, 0), -1)
                
                # 计算全局坐标
                global_x, global_y = x + px, y + py
                pupil_positions.append((global_x, global_y))
                
                # 在图像上显示瞳孔坐标
                cv2.putText(frame, f"Pupil: ({global_x}, {global_y})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 如果有检测到瞳孔，更新虚拟眼睛位置
        if pupil_positions:
            # 取第一个检测到的瞳孔位置
            pupil_x, pupil_y = pupil_positions[0]
            
            # 映射到虚拟眼睛坐标(150±50, 100±30)
            eye_x = 150 + (pupil_x - 320) / 6  # 640/2=320是画面水平中心
            eye_y = 100 + (pupil_y - 240) / 6  # 480/2=240是画面垂直中心
            self.eye_widget.set_pupil_position(eye_x, eye_y)
        
        # 显示摄像头画面
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.camera_view.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.stop_tracking()
        event.accept()

class PupilDetector:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.eye_params = {'scaleFactor': 1.1, 'minNeighbors': 8, 'minSize': (40, 40)}
        self.pupil_params = {'threshold': 30, 'min_area': 30, 'max_area': 1000}
        self.is_running = False
        self.cap = None
    
    def detect_pupil(self, eye_region):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        _, thresh = cv2.threshold(gray_eye, self.pupil_params['threshold'], 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        if not (self.pupil_params['min_area'] < area < self.pupil_params['max_area']):
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        return (int(cx), int(cy)), int(radius)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PupilDetectionApp()
    window.show()
    sys.exit(app.exec())