import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import QTimer, Qt, QRectF, QPointF


class EyeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 200)
        self.pupil_pos = QPointF(150, 100)  # 当前绘制位置
        self.target_pos = QPointF(150, 100)  # 目标位置
        self.pupil_radius = 20
        self.iris_radius = 40
        self.eye_color = QColor(200, 200, 255)  # 浅蓝色眼白
        self.iris_color = QColor(80, 120, 200)  # 蓝色虹膜

        # 定时器用于平滑更新
        self.smooth_timer = QTimer()
        self.smooth_timer.timeout.connect(self.smooth_update)
        self.smooth_timer.start(30)

    def set_pupil_position(self, x, y):
        x = max(80, min(220, x))  # 水平限制
        y = max(70, min(130, y))  # 垂直限制
        self.target_pos = QPointF(x, y)

    def smooth_update(self):
        smoothing_factor = 0.2
        new_x = self.pupil_pos.x() + (self.target_pos.x() - self.pupil_pos.x()) * smoothing_factor
        new_y = self.pupil_pos.y() + (self.target_pos.y() - self.pupil_pos.y()) * smoothing_factor
        self.pupil_pos = QPointF(new_x, new_y)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 眼白
        painter.setBrush(self.eye_color)
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawEllipse(QRectF(50, 50, 200, 100))

        # 虹膜
        painter.setBrush(self.iris_color)
        painter.drawEllipse(self.pupil_pos, self.iris_radius, self.iris_radius)

        # 瞳孔
        painter.setBrush(Qt.GlobalColor.black)
        painter.drawEllipse(self.pupil_pos, self.pupil_radius, self.pupil_radius)

        # 高光
        painter.setBrush(Qt.GlobalColor.white)
        highlight_pos = QPointF(self.pupil_pos.x() + 10, self.pupil_pos.y() - 10)
        painter.drawEllipse(highlight_pos, 5, 5)


class PupilDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pupil Tracking with Virtual Eye")
        self.setGeometry(100, 100, 1000, 600)

        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.cap = None

        # 主界面布局
        main_widget = QWidget()
        layout = QHBoxLayout()

        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)

        self.eye_widget = EyeWidget()

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
            self.cap = cv2.VideoCapture(1)
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
                    self.eye_widget.set_pupil_position(eye_x, eye_y)
                    pupil_detected = True

            # 显示摄像头画面
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_view.setPixmap(QPixmap.fromImage(qt_image))

            if not pupil_detected:
                # 如果没有检测到瞳孔，让虚拟眼睛回到中心位置
                self.eye_widget.set_pupil_position(150, 100)
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
    