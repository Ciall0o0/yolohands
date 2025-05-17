import torch
import os
from ultralytics import YOLO
from pathlib import Path
from ray import tune
# 检查可用设备，优先使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def raytune(model,base_path):
     # 定义基础路径和数据文件路径
    #base_path = Path("D:/Models/YoloFreiHAND/yolo")
    data_file = base_path / "train.yaml"
    # 加载预训练模型
    if model is None:
        model = YOLO(base_path / "yolo11n.pt")
    """
    使用Ray Tune进行模型超参数调优。
    """
    # 调用模型的tune方法进行超参数搜索
    result_grid = model.tune(
        data=data_file,          # 数据配置文件路径
        use_ray=True,             # 使用Ray Tune进行调优
        grace_period=30,        # 初始运行时间（秒）
        iterations=10,            # 迭代次数
        epochs=10,                # 每个试验的训练轮数
        batch=0.8,                # 批量大小比例,80% gpu
        optimizer="AdamW",        # 优化器类型
        save=True,             # 是否保存模型
    )

    # 输出每个试验的结果
    for i, result in enumerate(result_grid):
        print(f"试验#{i}: 配置: {result.config}, 最终指标: {result.metrics}")

