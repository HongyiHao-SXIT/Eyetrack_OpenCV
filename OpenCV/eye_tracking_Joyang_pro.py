import cv2
import pygame
import numpy as np

pygame.init()

screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pupil Tracking with Image Control")

image = pygame.image.load('img/eye.png')
image_rect = image.get_rect()

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)


prev_x, prev_y = 0, 0
alpha = 0.05
scale_factor = 4
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) > 0:
        (x, y, w, h) = eyes[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eye_region = gray[y:y + h, x:x + w]

        _, thresh = cv2.threshold(eye_region, 30, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)

            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

            scaled_x = cx * scale_factor - 250
            scaled_y = cy * (0.5 * scale_factor) - 1

            smoothed_x = prev_x + alpha * (scaled_x - prev_x)
            smoothed_y = prev_y + alpha * (scaled_y - prev_y)

            print(f"Smoothed pupil coordinates: x = {int(smoothed_x)}, y = {int(smoothed_y)}")

            prev_x, prev_y = smoothed_x, smoothed_y

            image_rect.center = (int(smoothed_x), int(smoothed_y))

    screen.fill((0, 0, 0))
    screen.blit(image, image_rect)
    pygame.display.flip()

    cv2.imshow('Pupil Detection', frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
pygame.quit()
