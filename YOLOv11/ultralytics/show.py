import cv2
from ultralytics import YOLO

# 加载训练好的模型
# 请将 'path/to/best.pt' 替换为你实际训练好的模型文件路径
# 假设训练好的模型在 runs/train/train-200epoch-v11n.yaml/weights/best.pt
model_path = 'YOLOv11\\runs\\train\\train-200epoch-v11n.yaml\\weights\\last.pt'
model = YOLO(model_path)

# 模型推理，使用电脑默认摄像头作为输入
cap = cv2.VideoCapture('img\\eye_data.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening camera")
    exit()

# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计算裁剪后的宽度和高度
crop_width = width // 2
crop_height = height // 2

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('runs/predict2/eye_data_detected.mp4', fourcc, fps, (crop_width, crop_height))

while cap.isOpened():
    # 读取一帧视频
    ret, frame = cap.read()

    if ret:
        # 裁剪画面为左上角的四分之一部分
        cropped_frame = frame[:crop_height, :crop_width]

        # 进行目标检测
        results = model(cropped_frame)

        # 获取检测结果
        for result in results:
            boxes = result.boxes  # 检测框信息
            if boxes is not None:
                for box in boxes:
                    class_id = box.cls.cpu().numpy()[0]
                    confidence = box.conf.cpu().numpy()[0]
                    bbox = box.xyxy.cpu().numpy()[0].astype(int)

                    # 打印检测结果
                    print(f"Class ID: {class_id}, Confidence: {confidence}, Bbox: {bbox}")

                    # 在裁剪后的帧上绘制检测框和标签
                    cv2.rectangle(cropped_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(cropped_frame, f"{int(class_id)}: {confidence:.2f}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示裁剪后的帧
        cv2.imshow('YOLOv11 Detection', cropped_frame)

        # 写入裁剪后的视频文件
        out.write(cropped_frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()