import supervision as sv
from pathlib import Path

def process(base_path: str):
    base_path = Path(base_path)
    
    # 从COCO格式加载训练集数据集
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=base_path,
        annotations_path=base_path / 'annotations/freihand_train.json',
    )
    
    # 从COCO格式加载测试集数据集
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=base_path,
        annotations_path=base_path / 'annotations/freihand_test.json',
    )
    
    # 从COCO格式加载验证集数据集
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=base_path,
        annotations_path=base_path / 'annotations/freihand_val.json',
    )
    
    # 打印原始数据集的类别信息和长度
    print(ds_train.classes)
    print(f"原始数据集大小 - 训练集: {len(ds_train)}, 验证集: {len(ds_valid)}, 测试集: {len(ds_test)}")
    
    # 将训练集按比例分割成新的训练集和缓存集
    ds_train, cache = ds_train.split(split_ratio=0.2, shuffle=True)
    
    # 将验证集按比例分割成新的验证集和缓存集
    ds_valid, cache1 = ds_valid.split(split_ratio=0.2, shuffle=True)
    
    # 将测试集按比例分割成新的测试集和缓存集
    ds_test, cache2 = ds_test.split(split_ratio=0.1, shuffle=True)
    
    # 打印分割后数据集的长度
    print(f"分割后的数据集大小 - 训练集: {len(ds_train)}, 验证集: {len(ds_valid)}, 测试集: {len(ds_test)}")
    
    # 将训练集转换为YOLO格式并保存
    ds_train.as_yolo(
        images_directory_path=base_path / 'yolo/train/images',
        annotations_directory_path=base_path / 'yolo/train/labels',
        data_yaml_path=base_path / 'yolo/train/train.yaml'
    )
    
    # 将验证集转换为YOLO格式并保存
    ds_valid.as_yolo(
        images_directory_path=base_path / 'yolo/val/images',
        annotations_directory_path=base_path / 'yolo/val/labels',
        data_yaml_path=base_path / 'yolo/val/val.yaml'
    )
    
    # 将测试集转换为YOLO格式并保存
    ds_test.as_yolo(
        images_directory_path=base_path / 'yolo/test/images',
        annotations_directory_path=base_path / 'yolo/test/labels',
        data_yaml_path=base_path / 'yolo/test/test.yaml'
    )
    
    print("数据处理完成")
