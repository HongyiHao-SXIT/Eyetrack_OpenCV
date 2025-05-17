import cv2
import pygame
import os
from threading import Thread
from yolo_utils import PupilDetector

cap = cv2.VideoCapture(2,cv2.CAP_DSHOW)

model_path = os.path.join("YOLOv11", "runs", "train", "train-200epoch-v11n.yaml", "weights", "last.pt")
detector = PupilDetector(model_path)

pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pupil Tracking Animation")

image_folder = "images"
images = [pygame.image.load(os.path.join(image_folder, f)) for f in os.listdir(image_folder) if f.endswith(".png")]
current_image_index = 0

clock = pygame.time.Clock()

pupil_x, pupil_y = 0.5, 0.5

def update_pupil_coords():
    global pupil_x, pupil_y
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        pupil_x, pupil_y = detector.detect_pupil_center(frame)

thread = Thread(target=update_pupil_coords, daemon=True)
thread.start()

running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                current_image_index = (current_image_index + 1) % len(images)

    img = images[current_image_index]
    img_rect = img.get_rect()
    img_w, img_h = img_rect.size
    center_x = int(pupil_x * screen_width) - img_w // 2
    center_y = int(pupil_y * screen_height) - img_h // 2

    screen.blit(img, (center_x, center_y))
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()
