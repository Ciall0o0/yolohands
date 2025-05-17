# full process
## pip install ray[default] ultralytics pathlib supervision openxlab wandb click==8.1.3
from pathlib import Path
import os
import model_train
import datasetpro
import model_tune
import model_tune_ray
import shutil
base_path=Path("D:/Models/YoloFreiHAND/")
## 1. 数据集下载
import openxlab
def download(ak:str,sk:str):
    openxlab.login(ak, sk)
    from openxlab.dataset import get
    get(dataset_repo='OpenDataLab/FreiHAND', target_base_path=base_path) # 数据集下载
    pass

def preprocess_data():
    # 数据预处理
    datasetpro.process(base_path)
    pass

def tune_and_train(model,useRay:bool):
    if useRay == True:
        model_tune_ray.raytune(model,base_path/"yolo")
    else:
        model_tune.tune(model,base_path/"yolo")
    model_train.train(base_path/"yolo")
    pass

def search_files(directory, pattern):
        matches = []
        for root, _, files in os.walk(directory):
            for file in fnmatch.filter(files, pattern):
                matches.append(os.path.join(root, file))
        return matches

def get_latest_file(matches):
        if not matches:
            return None
        latest_file = max(matches, key=os.path.getmtime)
        return latest_file

directory =base_path / "runs/detect/"
pattern = 'best.pt'
matches = search_files(directory, pattern)
latest_file = get_latest_file(matches)

def streamlit():
    if latest_file:
        try:
            shutil.copy(latest_file, base_path)
        except Exception as e:
            print(f"Error copying file: {e}")
    os.system("streamlit run " + str(base_path/"yolo"/"stream.py"))
    pass

if __name__ == "__main__":
    #download("your_ak", "your_sk")
    #preprocess_data()
    model = None
    #tune_and_train(model, useRay=False)
    streamlit()