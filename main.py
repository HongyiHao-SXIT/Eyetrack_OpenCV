import cv2

image = cv2.imread('Lanyi_adict.jpg')

if image is not None:
    # 显示图片
    cv2.imshow('Image', image)
    # 等待按键事件
    cv2.waitKey(0)
    # 关闭所有窗口
    cv2.destroyAllWindows()
else:
    print('无法读取图片，请检查图片路径。')