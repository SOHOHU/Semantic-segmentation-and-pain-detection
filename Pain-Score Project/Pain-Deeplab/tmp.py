import json
import os
import shutil

# 设置 A 和 B 文件夹的路径
folder_a = 'datasets/before'
folder_b = r'C:\Users\23600\OneDrive\文档\硕士\奶牛\代码\video2jpg\jpg\stress'

# 遍历 A 文件夹中的所有 JSON 文件
for filename in os.listdir(folder_a):
    if filename.endswith('.json'):
        # 构建 JSON 文件名对应的 JPG 文件名
        jpg_name = filename.replace('.json', '.jpg')

        # 构建 JPG 文件的完整路径
        jpg_path = os.path.join(folder_b, jpg_name)
        if os.path.exists(jpg_path):
            # 如果 JPG 文件存在，则复制到 A 文件夹
            shutil.copy(jpg_path, folder_a)
        else:
            print(f'文件 {jpg_name} 不存在于 {folder_b} 中')
