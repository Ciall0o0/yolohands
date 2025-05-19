### Yolohands
### 本人大一小登，欢迎各位大佬指正！
![miku](https://img.picui.cn/free/2025/05/19/682ae90fc47ce.gif)

# 1.下载数据集
OpenDataLab/FreiHAND:[https://opendatalab.com/OpenDataLab/FreiHAND/tree/main](https://opendatalab.com/OpenDataLab/FreiHAND/tree/main)
# 2.处理数据集
使用[supersivion](https://supervision.roboflow.com/latest/how_to/process_datasets)将数据集处理成YOLO格式
# 3.超参数调整
常规方法:定义搜索空间(详见ultralytics的[超参数调整方法](https://docs.ultralytics.com/guides/hyperparameter-tuning/)),例如:
```
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
```
结果:
```
# 5/5 iterations complete ✅ (113.93s)
# Results saved to runs\detect\tune
# Best fitness=0.0 observed at iteration 1
# Best fitness metrics are {}
# Best fitness model is runs\detect\train
# Best fitness hyperparameters are printed below.

lr0: 0.01
degrees: 0.0
weight_decay: 0.0005
momentum: 0.937
box: 7.5
cls: 0.5
dfl: 1.5
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.4
translate: 0.1
scale: 0.5
```
使用ray tune进行超参数调整,需要安装ray[tune]库

ray tune: 参考[https://docs.ray.io/en/latest/tune/index.html](https://docs.ray.io/en/latest/tune/index.html)和
[https://docs.ultralytics.com/integrations/ray-tune/](https://docs.ultralytics.com/integrations/ray-tune/)
# 4.训练
参数解释:[https://blog.csdn.net/qq_37553692/article/details/130898732](https://blog.csdn.net/qq_37553692/article/details/130898732)
```
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
```

结果:(25 epochs tune+100 epochs train)

![results.png](https://github.com/Ciall0o0/yolohands/blob/master/runs/detect/train13/results.png)
# 5.推理
使用[streamlit](https://streamlit.io/)创建一个简单的web应用,可参考[https://docs.ultralytics.com/zh/guides/streamlit-live-inference/#streamlit-application-code](https://docs.ultralytics.com/zh/guides/streamlit-live-inference/#streamlit-application-code)
在ultralytics库中包含一个基础演示,我改动了一部分,增添了fps显示与一些处理机制,效果如下: 
![截图](https://github.com/Ciall0o0/yolohands/blob/master/屏幕截图.png)

## 使用方法
```
git clone https://github.com/Ciall0o0/yolohands.git
cd yolohands
uv sync
```
有关uv使用,参考[https://zhuanlan.zhihu.com/p/1888904532131575259](https://zhuanlan.zhihu.com/p/1888904532131575259)
配置好环境后,修改fullprocess.py中的base_path,然后
```
uv run fullprocess.py
```
or
```
python fullprocess.py
```
