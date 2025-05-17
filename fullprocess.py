import os
import shutil
from pathlib import Path
import fnmatch
import openxlab
import model_train
import datasetpro
import model_tune
import model_tune_ray

# 设置基础路径
base_path = Path("D:/Models/YoloFreiHAND/")

def download(ak: str, sk: str):
    """
    使用OpenXLab下载FreiHAND数据集。
    
    :param ak: Access Key
    :param sk: Secret Key
    """
    openxlab.login(ak, sk)
    from openxlab.dataset import get
    get(dataset_repo='OpenDataLab/FreiHAND', target_base_path=base_path)

def preprocess_data():
    """
    对下载的数据集进行预处理。
    """
    datasetpro.process(base_path)

def tune_and_train(model, useRay: bool):
    """
    调整模型超参数并训练模型。
    
    :param model: 模型对象
    :param useRay: 是否使用Ray进行调优
    """
    if useRay:
        model_tune_ray.raytune(model, base_path / "yolo")
    else:
        model_tune.tune(model, base_path / "yolo")
    model_train.train(base_path / "yolo")

def search_files(directory: Path, pattern: str):
    """
    在指定目录中搜索符合模式的文件。
    
    :param directory: 目录路径
    :param pattern: 文件名模式
    :return: 符合模式的文件列表
    """
    matches = []
    for root, _, files in os.walk(directory):
        for file in fnmatch.filter(files, pattern):
            matches.append(os.path.join(root, file))
    return matches

def get_latest_file(matches: list):
    """
    从匹配的文件列表中获取最新的文件。
    
    :param matches: 匹配的文件列表
    :return: 最新的文件路径，如果没有找到则返回None
    """
    if not matches:
        return None
    latest_file = max(matches, key=os.path.getmtime)
    return latest_file

def streamlit():
    """
    运行Streamlit应用。
    """
    directory = base_path / "runs/detect/"
    pattern = 'best.pt'
    matches = search_files(directory, pattern)
    latest_file = get_latest_file(matches)
    
    if latest_file:
        try:
            # 将最佳权重文件复制到基础路径
            shutil.copy(latest_file, base_path)
        except Exception as e:
            print(f"Error copying file: {e}")
    
    # 启动Streamlit应用
    os.system(f"streamlit run {str(base_path/'yolo'/'stream.py')}")

if __name__ == "__main__":
    # 下载数据集（需要提供Access Key和Secret Key）
    download("your_ak", "your_sk")
    
    # 预处理数据
    preprocess_data()
    
    # 定义模型对象（这里假设为None，实际应传入具体模型）
    model = None
    
    # 调整超参数并训练模型（可以选择是否使用Ray）
    tune_and_train(model, useRay=False)
    
    # 运行Streamlit应用
    streamlit()
