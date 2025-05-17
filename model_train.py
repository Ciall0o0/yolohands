import torch
from ultralytics import YOLO
import os
# 检查可用设备，优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置批量大小比例和工作线程数
batch_size = -1 ## 自动
workers = 4

def train(base_path):
    """
    训练YOLO模型并将其导出为TensorRT格式。
    """
    # 定义基础路径和数据文件路径
    #base_path = Path("D:/Models/YoloFreiHAND/yolo")
    data_file = base_path / "train.yaml"
    # 加载预训练模型
    if os.path.exists(base_path / "runs/detect/tune/weights/best.pt"):
        model = YOLO(base_path / "runs/detect/tune/weights/best.pt")
    else:
        print("没有找到tune后权重文件，使用默认模型")
        model = YOLO(base_path / "yolo11n.pt")

    # 开始训练模型
    model.train(
        data=data_file,           # 数据配置文件路径
        epochs=20,               # 训练轮数
        imgsz=512,                # 图像尺寸
        device=device,            # 计算设备
        pretrained=True,          # 使用预训练权重
        save=True,                # 保存最佳模型
        patience=5,              # 早停耐心值
        project="train",          # 项目名称
        amp=True,                 # 自动混合精度
        cos_lr=True,              # 余弦退火学习率调度器
        batch=batch_size,         # 批量大小比例
        workers=workers,          # 工作线程数
        close_mosaic=True,        # 关闭mosaic增强
    )
    pass
