@echo off
REM DeeplabV3+ 优化版训练脚本 (Windows)
REM 使用方法: 双击运行或在命令行执行 train_optimized.bat

echo ========================================
echo DeeplabV3+ 优化版训练脚本
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

REM 检查CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  未检测到CUDA设备，将使用CPU训练(速度较慢^)
    echo.
) else (
    echo ✅ 检测到CUDA设备:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo.
)

REM 检查必要的依赖
echo 检查依赖库...
python -c "import torch; import torchvision; import cv2; import numpy" 2>nul
if errorlevel 1 (
    echo ❌ 缺少必要的依赖库
    echo 正在安装依赖...
    pip install -r requirements.txt
)

REM 检查数据集
if not exist "VOCdevkit\VOC2007\JPEGImages" (
    echo ⚠️  警告: 未找到数据集目录 VOCdevkit\VOC2007\JPEGImages
    echo 请确保数据集已正确放置
    set /p continue="是否继续? (y/n): "
    if not "%continue%"=="y" exit /b 1
)

REM 创建日志目录
if not exist "logs" mkdir logs
if not exist "model_data" mkdir model_data

echo.
echo ========================================
echo 训练配置
echo ========================================
echo ✅ 高级数据增强: 启用
echo ✅ OHEM损失: 启用
echo ✅ 边界损失: 启用
echo ✅ 模型EMA: 启用
echo ✅ FP16混合精度: 启用
echo ✅ 训练周期: 300 epochs
echo.

pause

REM 开始训练
echo.
echo ========================================
echo 开始训练...
echo ========================================
echo.

python train.py

REM 训练完成
echo.
echo ========================================
echo ✅ 训练完成!
echo ========================================
echo.
echo 训练结果保存在 logs\ 目录
echo 最佳模型: logs\best_epoch_weights_ema.pth (推荐)
echo.
echo 下一步:
echo 1. 评估模型: python get_miou.py
echo 2. 进行推理: python predict.py
echo 3. 性能对比: python performance_compare.py
echo.

pause

