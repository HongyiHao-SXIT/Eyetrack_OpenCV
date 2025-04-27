import pygame
import random

pygame.init()

screen_width, screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.NOFRAME | pygame.FULLSCREEN)

image = pygame.image.load('img\eye.png')
image_width, image_height = image.get_size()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # 当按下Esc键时退出程序
                running = False

    try:
        # 从文件中读取坐标
        with open('pupil_coordinates.txt', 'r') as file:
            coordinates = file.read().strip().split(',')
            x, y = int(coordinates[0]), int(coordinates[1])
    except (FileNotFoundError, IndexError):
        # 如果文件不存在或格式错误，使用随机初始位置
        x, y = random.randint(0, screen_width - image_width), random.randint(0, screen_height - image_height)

    screen.fill((0, 0, 0))
    screen.blit(image, (x, y))
    pygame.display.flip()
    pygame.time.Clock().tick(60)

pygame.quit()