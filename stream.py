#from ultralytics import solutions
import stream_deploy
inf = stream_deploy.Inference(
        model="best.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
    )
inf.inference()
# Make sure to run the file using command `streamlit run path/to/file.py`