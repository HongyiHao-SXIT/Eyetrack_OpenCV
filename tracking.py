import cv2
import numpy as np
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture('img/eyes_data.mp4')
#cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eye_region = gray[y:y + h, x:x + w]
        eye_frame = frame[y:y + h, x:x + w]

        _, thresh = cv2.threshold(eye_region, 20, 255, cv2.THRESH_BINARY_INV) 
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if cv2.contourArea(c) > 25]
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            
            cv2.circle(eye_frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
            
            cv2.circle(eye_frame, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            
            print(f"Pupil coordinates: x={int(cx)}, y={int(cy)}")

            print(f"Pupil coordinates now: x={1.3 * int(cx)}, y={1.2 * int(cy)}")

    cv2.imshow('Pupil Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()