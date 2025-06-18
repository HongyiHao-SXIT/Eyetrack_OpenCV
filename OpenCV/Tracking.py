import cv2
import numpy as np

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cropped_width = width // 2
cropped_height = height // 2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (cropped_width, cropped_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cropped_frame = frame[:cropped_height, :cropped_width]

    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eye_region = gray[y:y + h, x:x + w]
        eye_frame = cropped_frame[y:y + h, x:x + w]
        _, thresh = cv2.threshold(eye_region, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50 and area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            (cx, cy), radius = cv2.minEnclosingCircle(max_contour)
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)

            print(f"Pupil coordinates: x={int(cx)}, y={int(cy)}")

    cv2.imshow('Pupil Detection', cropped_frame)

    out.write(cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()