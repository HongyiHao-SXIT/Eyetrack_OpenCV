import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import QTimer, Qt, QRectF, QPointF


class PupilDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pupil Tracking with Image")
        self.setGeometry(100, 100, 1000, 600)

        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.cap = None

        # 初始化卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1e-4, 0, 0, 0], [0, 1e-4, 0, 0], [0, 0, 1e-4, 0], [0, 0, 0, 1e-4]], np.float32)
        self.kalman.measurementNoiseCov = np.array([[1e-1, 0], [0, 1e-1]], np.float32)

        self.prev_pupil_pos = None
        self.predicted_pupil_pos = None

        # 加载眼睛图片
        self.eye_image = cv2.imread('img/Lanyi_adict.jpg')
        if self.eye_image is None:
            raise Exception("无法加载眼睛图片，请检查路径。")
        self.eye_image = cv2.cvtColor(self.eye_image, cv2.COLOR_BGR2RGB)
        self.eye_height, self.eye_width, _ = self.eye_image.shape

        # 主界面布局
        main_widget = QWidget()
        layout = QHBoxLayout()

        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)

        self.eye_widget = QLabel()
        self.eye_widget.setFixedSize(300, 200)

        control_panel = QWidget()
        control_layout = QVBoxLayout()

        self.start_btn = QPushButton("Start Tracking")
        self.stop_btn = QPushButton("Stop Tracking")
        self.exit_btn = QPushButton("Exit")

        button_style = """
        QPushButton {
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 5px;
            min-width: 120px;
        }
        """
        self.start_btn.setStyleSheet(button_style + "background-color: #4CAF50; color: white;")
        self.stop_btn.setStyleSheet(button_style + "background-color: #F44336; color: white;")
        self.exit_btn.setStyleSheet(button_style + "background-color: #607D8B; color: white;")

        self.stop_btn.setEnabled(False)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 12))

        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.exit_btn)
        control_layout.addStretch()
        control_panel.setLayout(control_layout)

        layout.addWidget(self.camera_view)
        layout.addWidget(self.eye_widget)
        layout.addWidget(control_panel)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.timer = QTimer()
        self.is_tracking = False

        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.exit_btn.clicked.connect(self.close)

    def start_tracking(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头，请检查设备连接。")
            self.is_tracking = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Status: Tracking...")

            self.timer.timeout.connect(self.process_frame)
            self.timer.start(30)  # 约30fps
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.status_label.setText("Status: Error - " + str(e))

    def stop_tracking(self):
        self.is_tracking = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.camera_view.clear()
        self.eye_widget.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")

    def process_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("无法读取摄像头帧，请检查设备。")
            frame = cv2.flip(frame, 1)  # 水平翻转使画面更自然
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 眼睛检测
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            pupil_detected = False
            for (x, y, w, h) in eyes:
                # 绘制眼睛矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 提取眼睛区域
                eye_region = gray[y:y + h, x:x + w]
                eye_frame = frame[y:y + h, x:x + w]

                # 瞳孔检测
                _, thresh = cv2.threshold(eye_region, 20, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 只处理面积大于25的轮廓
                valid_contours = [c for c in contours if cv2.contourArea(c) > 25]

                if valid_contours:
                    # 找到面积最大的轮廓(主瞳孔)
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                    # 绘制瞳孔标记
                    cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
                    cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)

                    # 映射到虚拟眼睛坐标
                    eye_x = 150 + (cx - w / 2) * 0.5  # 缩放因子使移动更平滑
                    eye_y = 100 + (cy - h / 2) * 0.5

                    # 更新卡尔曼滤波器
                    measurement = np.array([[np.float32(eye_x)], [np.float32(eye_y)]])
                    if self.prev_pupil_pos is None:
                        self.kalman.statePre = np.array([[eye_x], [eye_y], [0], [0]], np.float32)
                        self.kalman.statePost = np.array([[eye_x], [eye_y], [0], [0]], np.float32)
                    else:
                        self.kalman.correct(measurement)
                        self.predicted_pupil_pos = self.kalman.predict()
                        eye_x = self.predicted_pupil_pos[0][0]
                        eye_y = self.predicted_pupil_pos[1][0]

                    self.prev_pupil_pos = (eye_x, eye_y)
                    pupil_detected = True

                    # 根据瞳孔位置调整眼睛图片位置
                    x_offset = int(eye_x - self.eye_width / 2)
                    y_offset = int(eye_y - self.eye_height / 2)
                    x_offset = max(0, min(300 - self.eye_width, x_offset))
                    y_offset = max(0, min(200 - self.eye_height, y_offset))

                    qt_image = QImage(self.eye_image.data, self.eye_width, self.eye_height, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.eye_widget.setPixmap(pixmap)
                    self.eye_widget.move(x_offset, y_offset)

            # 显示摄像头画面
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_view.setPixmap(QPixmap.fromImage(qt_image))

            if not pupil_detected:
                # 如果没有检测到瞳孔，让虚拟眼睛回到中心位置
                self.eye_widget.move(150 - self.eye_width / 2, 100 - self.eye_height / 2)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.stop_tracking()
            self.status_label.setText("Status: Error - " + str(e))

    def closeEvent(self, event):
        self.stop_tracking()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PupilDetectionApp()
    window.show()
    sys.exit(app.exec())