import torch
from ultralytics import YOLO
from pathlib import Path

# 检查可用设备，优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 扩展搜索空间
search_space = {
    "lr0": (1e-5, 1e-1),          # 初始学习率范围
    "degrees": (0.0, 45.0),       # 旋转角度范围
    "weight_decay": (1e-5, 1e-2), # 权重衰减范围
    "momentum": (0.8, 0.99),      # 动量范围
    "box": (1.0, 10.0),           # 边界框损失权重范围
    "cls": (0.1, 1.0),            # 分类损失权重范围
    "dfl": (0.5, 2.0),            # DFL损失权重范围
    "hsv_h": (0.0, 0.1),          # HSV色调扰动范围
    "hsv_s": (0.0, 0.5),          # HSV饱和度扰动范围
    "hsv_v": (0.0, 0.5),          # HSV明度扰动范围
    "translate": (0.0, 0.5),      # 平移扰动范围
    "scale": (0.5, 2.0),          # 缩放扰动范围
}

batch_size = 0.80
workers = 4

def tune(model,base_path):
    # 定义基础路径和数据文件路径
    #base_path = Path("D:/Models/YoloFreiHAND/yolo")
    data_file = base_path / "train.yaml"
    # 加载预训练模型
    if model is None:
        model = YOLO(base_path / "yolo11n.pt")

    # 进行超参数调优
    model.tune(
        data=data_file,           # 数据配置文件路径
        epochs=5,                 # 每个试验的训练轮数
        iterations=5,            # 迭代次数
        imgsz=384,                # 图像尺寸
        optimizer="AdamW",        # 优化器类型
        space=search_space,       # 超参数搜索空间
        plots=True,               # 是否生成绘图
        save=True,                # 保存最佳模型
        val=True,                # 进行验证集评估
    )