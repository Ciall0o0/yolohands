#from ultralytics import solutions
import stream_deploy
def streamlit_app():
    inf = stream_deploy.Inference(
        model="yolo11n.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
    )
    inf.inference()

# Make sure to run the file using command `streamlit run path/to/file.py`