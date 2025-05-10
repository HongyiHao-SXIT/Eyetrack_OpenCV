#yolo 封装

# # yolo_utils.py
# from ultralytics import YOLO
# import numpy as np

# class PupilDetector:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     def detect_pupil_center(self, frame):
#         results = self.model.predict(source=frame, verbose=False, conf=0.5)
#         if len(results) == 0 or results[0].boxes is None:
#             return 0.5, 0.5  # 默认画面中心

#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         if len(boxes) == 0:
#             return 0.5, 0.5

#         # 取第一个框作为瞳孔（你也可以改为置信度最大）
#         x1, y1, x2, y2 = boxes[0]
#         center_x = (x1 + x2) / 2
#         center_y = (y1 + y2) / 2

#         h, w, _ = frame.shape
#         return center_x / w, center_y / h

# yolo_utils.py
from ultralytics import YOLO
import numpy as np

class PupilDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_pupil_center(self, frame):
        results = self.model.predict(source=frame, verbose=False, conf=0.3)  # 初始较低阈值保留框
        if len(results) == 0 or results[0].boxes is None:
            return 0.5, 0.5  # 默认画面中心

        boxes = results[0].boxes
        if boxes.conf is None or len(boxes.conf) == 0:
            return 0.5, 0.5

        # 取第一个满足置信度条件的框
        for i, conf in enumerate(boxes.conf.cpu().numpy()):
            if conf >= 0.80:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                h, w, _ = frame.shape
                return center_x / w, center_y / h

        return 0.5, 0.5  # 若无框满足条件，返回中心

