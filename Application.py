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


class PupilDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("瞳孔识别系统 (QT6)")
        self.setGeometry(100, 100, 800, 600)

        # 主控件和布局
        self.main_widget = QWidget()
        self.layout = QVBoxLayout()

        # 视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)

        # 按钮
        self.start_btn = QPushButton("开始识别")
        self.stop_btn = QPushButton("停止识别")
        self.exit_btn = QPushButton("退出程序")

        # 设置按钮样式
        self.start_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        self.stop_btn.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 8px;"
        )
        self.exit_btn.setStyleSheet(
            "background-color: #607D8B; color: white; font-weight: bold; padding: 8px;"
        )

        # 初始禁用停止按钮
        self.stop_btn.setEnabled(False)

        # 添加控件到布局
        self.layout.addWidget(self.video_label)
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)
        self.layout.addWidget(self.exit_btn)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        # 初始化摄像头和定时器
        self.cap = None
        self.timer = QTimer()
        self.is_detecting = False

        # 连接信号
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.exit_btn.clicked.connect(self.close)

    def start_detection(self):
        """启动瞳孔识别"""
        self.cap = cv2.VideoCapture(0)  # 默认摄像头
        if not self.cap.isOpened():
            print("错误: 无法打开摄像头")
            return

        self.is_detecting = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 更新一帧

    def stop_detection(self):
        """停止识别"""
        self.is_detecting = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self):
        """处理每一帧进行瞳孔检测"""
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        # 镜像翻转
        frame = cv2.flip(frame, 1)

        # 定义ROI (可调整)
        roi = frame[100:500, 200:700]

        # 图像处理
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        # 绘制瞳孔位置
        for cnt in contours[:1]:  # 只处理最大的一个轮廓
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(
                roi,
                (x + int(w / 2), 0),
                (x + int(w / 2), roi.shape[0]),
                (0, 255, 0),
                2,
            )
            cv2.line(
                roi,
                (0, y + int(h / 2)),
                (roi.shape[1], y + int(h / 2)),
                (0, 255, 0),
                2,
            )

        # 转换为QT可显示的格式
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PupilDetectionApp()
    window.show()
    sys.exit(app.exec())