import cv2
import numpy as np
import os


# 定义类别及其对应的颜色（RGB格式）
color_map = {
    "_background_": (0, 0, 0),      # 黑色
    "ear": (255, 0, 0),             # 红色
    "eye": (0, 255, 0),             # 绿色
    "mouth": (0, 0, 255),           # 蓝色
    "nose": (255, 255, 0),          # 黄色
    "iris": (0, 255, 255),          # 青色
    "saliva": (255, 0, 255)         # 品红色
}

def gray2rgb(gray_path, save_path):
    # 读取灰度图像
    gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

    # 创建一个空白的RGB图像用于上色
    color_image = np.zeros((*gray_image.shape, 3), dtype=np.uint8)

    # 上色
    for i in range(len(color_map)):
        color_image[gray_image == i] = color_map[list(color_map.keys())[i]]

    # 保存上色后的图像
    cv2.imwrite(save_path, color_image)

if __name__ == "__main__":
    gray_path = "miou_out/detection-results"
    save_path = "miou_out/rgb-results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for file in os.listdir(gray_path):
        gray2rgb(os.path.join(gray_path, file), os.path.join(save_path, file))
