#!/bin/bash

# 快速提取面部部件脚本 (Linux/Mac) - 在项目根目录运行

echo "================================================================================"
echo "                        面部部件提取工具"
echo "================================================================================"
echo ""
echo "本工具将从分割结果中提取各个面部部件并单独保存到项目根目录"
echo ""
echo "提取的部件:"
echo "  - face (脸部)"
echo "  - mouth (嘴部)"
echo "  - nose (鼻子)"
echo "  - eyes (眼睛)"
echo "  - ear (耳朵)"
echo "  - muscles_above_eye (眉毛上方肌肉)"
echo ""
echo "输出位置: 项目根目录/face_parts_extracted/"
echo ""
echo "================================================================================"
echo ""

cd "Deeplab系列/deeplab"
python extract_face_parts.py

