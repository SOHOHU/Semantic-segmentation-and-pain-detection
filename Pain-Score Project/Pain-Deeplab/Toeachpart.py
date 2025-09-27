import json
import cv2
import numpy as np
import os

input_json_dir = r'D:\KC_Code\VOC2007\Normal'
output_root_dir = r'D:\KC_Code\VOC2007\TrainPain\result'

# 创建输出根目录（如果不存在）
os.makedirs(output_root_dir, exist_ok=True)

for filename in os.listdir(input_json_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(input_json_dir, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_width = data['imageWidth']
        image_height = data['imageHeight']

        canvas = np.zeros((image_height, image_width), dtype=np.uint8)

        for shape in data['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)

            # 绘制多边形
            cv2.fillPoly(canvas, [points], (255, 255, 255))

            # 创建子目录
            sub_dir = os.path.join(output_root_dir, label)
            os.makedirs(sub_dir, exist_ok=True)

            # 生成输出路径
            output_path = os.path.join(sub_dir, f'{os.path.splitext(filename)[0]}_{label}_mask.png')

            # 保存图像
            cv2.imwrite(output_path, canvas)

            # 清空画布
            canvas.fill(0)

        print(f"Processed: {filename}")