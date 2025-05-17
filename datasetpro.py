import supervision as sv
from pathlib import Path
def process(base_path: str):
    #base_path = Path(base_path)
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=base_path,
        annotations_path=base_path / 'annotations/freihand_train.json',
    )
    ds_test= sv.DetectionDataset.from_coco(
        images_directory_path=base_path,
        annotations_path=base_path / 'annotations/freihand_test.json',
    )
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=base_path,
        annotations_path=base_path / 'annotations/freihand_val.json',
    )
    ds_train.classes
    print(len(ds_train),len(ds_valid),len(ds_test))
    ds_train, cache = ds_train.split(split_ratio=0.2, shuffle=True)
    ds_valid, cache1 = ds_valid.split(split_ratio=0.2, shuffle=True)
    ds_test, cache2 = ds_test.split(split_ratio=0.1, shuffle=True)
    print(len(ds_train),len(ds_valid),len(ds_test))
    ds_train.as_yolo(
        images_directory_path=base_path / 'yolo/train/images',
        annotations_directory_path=base_path / 'yolo/train/labels',
        data_yaml_path=base_path / 'yolo/train/train.yaml'
    )
    ds_valid.as_yolo(
        images_directory_path=base_path / 'yolo/val/images',
        annotations_directory_path=base_path / 'yolo/val/labels',
        data_yaml_path=base_path / 'yolo/val/val.yaml'
    )
    ds_test.as_yolo(
        images_directory_path=base_path / 'yolo/test/images',
        annotations_directory_path=base_path / 'yolo/test/labels',
        data_yaml_path=base_path / 'yolo/test/test.yaml'
    )
    print("finished")

    
