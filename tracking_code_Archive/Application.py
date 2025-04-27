import sys
import cv2
import numpy as np
import math
from collections import deque
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import QTimer


class EyeTrackingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.image_path = ""
        self.original_center = np.array([220, 180], dtype=np.float32)  # 图片初始中心位置
        self.return_speed = 0.03  # 回中速度，调小以让回中更缓慢

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.start_button = QPushButton('开始', self)
        self.start_button.setIcon(QIcon('start_icon.png'))
        self.start_button.clicked.connect(self.start_tracking)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.pause_button = QPushButton('暂停', self)
        self.pause_button.setIcon(QIcon('pause_icon.png'))
        self.pause_button.clicked.connect(self.pause_tracking)
        self.pause_button.setDisabled(True)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)

        self.exit_button = QPushButton('退出', self)
        self.exit_button.setIcon(QIcon('exit_icon.png'))
        self.exit_button.clicked.connect(self.close)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        self.select_image_button = QPushButton('选择图片', self)
        self.select_image_button.setIcon(QIcon('select_image_icon.png'))
        self.select_image_button.clicked.connect(self.select_image)
        self.select_image_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)

        self.cropped_label = QLabel(self)
        self.cropped_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken)
        self.cropped_label.setMinimumSize(300, 200)

        self.display_label = QLabel(self)
        self.display_label.setFrameStyle(QLabel.Shape.Box | QLabel.Shadow.Sunken)
        self.display_label.setMinimumSize(300, 200)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.exit_button)
        button_layout.addWidget(self.select_image_button)

        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.cropped_label)
        main_layout.addWidget(self.display_label)

        self.setLayout(main_layout)
        self.setWindowTitle('眼动追踪程序')
        self.setWindowIcon(QIcon('app_icon.png'))
        self.setGeometry(100, 100, 800, 600)

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', '图片文件 (*.jpg *.png)')
        if file_path:
            self.image_path = file_path

    def start_tracking(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头，请检查设备连接。")
                return
            self.timer.start(30)
            self.is_running = True
            self.start_button.setDisabled(True)
            self.pause_button.setDisabled(False)

    def pause_tracking(self):
        if self.is_running:
            self.timer.stop()
            self.is_running = False
            self.start_button.setDisabled(False)
            self.pause_button.setDisabled(True)
            if self.cap:
                self.cap.release()

    def update_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                miss_num = 0
                close_my_eye_num = 0
                position_change_threshold = 30
                SMOOTHING_WINDOW_SIZE = 5
                POSITION_HISTORY = deque(maxlen=SMOOTHING_WINDOW_SIZE)
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                display_width, display_height = 440, 360
                display_img = np.zeros((display_height, display_width, 3), dtype=np.uint8)

                if not self.image_path:
                    QMessageBox.critical(self, "错误", "未添加图片，请先选择图片文件。")
                    self.pause_tracking()
                    return

                controlled_img = cv2.imread(self.image_path)
                if controlled_img is None:
                    QMessageBox.critical(self, "错误", "无法找到图片，请检查图片路径。")
                    self.pause_tracking()
                    return

                controlled_img = cv2.resize(controlled_img, (50, 50))

                controlled_pos = np.array([display_width // 2, display_height // 2], dtype=np.float32)
                prev_pupil_positions = []

                def apply_smoothing(new_pos):
                    POSITION_HISTORY.append(new_pos)
                    if len(POSITION_HISTORY) < 2:
                        return new_pos
                    weights = np.linspace(0.1, 1.0, len(POSITION_HISTORY))
                    weights /= weights.sum()
                    smoothed_pos = np.zeros(2)
                    for i, pos in enumerate(POSITION_HISTORY):
                        smoothed_pos += pos * weights[i]
                    return smoothed_pos

                def interpolate_position(current_pos, target_pos, factor=0.2):
                    return current_pos * (1 - factor) + target_pos * factor

                height, width = frame.shape[:2]
                cropped_frame = frame[0:height // 2, 0:width // 2]
                gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

                eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                current_pupil_positions = []
                pupil_detected = False

                for (x, y, w, h) in eyes:
                    cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    eye_region = gray[y:y + h, x:x + w]
                    eye_frame = cropped_frame[y:y + h, x:x + w]
                    _, thresh = cv2.threshold(eye_region, 20, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
                    if valid_contours:
                        pupil_detected = True
                        largest_contour = max(valid_contours, key=cv2.contourArea)
                        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                        global_cx, global_cy = int(x + cx), int(y + cy)
                        current_pupil_positions.append((global_cx, global_cy))
                        cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
                        cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
                        print(f"Pupil coordinates: x={global_cx}, y={global_cy}")

                if not pupil_detected:
                    print("close_my_eye")
                    close_my_eye_num += 1
                    # 图片慢慢回中
                    controlled_pos = interpolate_position(controlled_pos, self.original_center, self.return_speed)
                elif prev_pupil_positions and current_pupil_positions:
                    position_changes = [
                        math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
                        for prev, curr in zip(prev_pupil_positions, current_pupil_positions)
                    ]
                    if any(change > position_change_threshold for change in position_changes):
                        print("miss")
                        miss_num += 1

                prev_pupil_positions = current_pupil_positions

                display_img[:] = (240, 240, 240)
                if current_pupil_positions:
                    pupil_x, pupil_y = current_pupil_positions[0]
                    target_x = np.clip(int(pupil_x * display_width / (width // 2)), 25, display_width - 25)
                    target_y = np.clip(int(pupil_y * display_height / (height // 2)), 25, display_height - 25)
                    target_pos = np.array([target_x, target_y], dtype=np.float32)
                    smoothed_target = apply_smoothing(target_pos)
                    controlled_pos = interpolate_position(controlled_pos, smoothed_target)

                draw_pos = controlled_pos.astype(int)
                x, y = draw_pos[0], draw_pos[1]
                display_img[y - 25:y + 25, x - 25:x + 25] = controlled_img

                cv2.rectangle(display_img, (0, 0), (display_width - 1, display_height - 1), (0, 0, 0), 2)
                cv2.putText(display_img, "11.0cm x 9.0cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                self.show_image(cropped_frame, self.cropped_label)
                self.show_image(display_img, self.display_label)

    def show_image(self, image, label):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(bytes(image.data), width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EyeTrackingApp()
    ex.show()
    sys.exit(app.exec())
    