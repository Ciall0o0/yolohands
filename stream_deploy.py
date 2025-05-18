# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import os
from typing import Any, List
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
import os
import torch
import time
import warnings
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
ASSETS_NAMES = frozenset(
        [f"best.pt"]
        +[f"last.pt"]
        + [f"yolo11{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb")]
    )
ASSETS_STEMS = frozenset(k.rsplit(".", 1)[0] for k in ASSETS_NAMES)

class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video files, and performing
    real-time inference using Streamlit and Ultralytics YOLO models.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds

        self.st = st  # Reference to the Streamlit module
        self.source = None  # Video source selection (webcam or video file)
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.25  # Confidence threshold for detection
        self.iou = 0.45  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.vid_file_name = None  # Video file name or webcam index
        self.selected_ind = []  # List of selected class indices for detection
        self.model = None  # YOLO model instance
        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # Model file path
        self.fps = 0
        self.prev_time = 0
        self.fps_display = None

        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]
        if "processed_frames" not in st.session_state:
            st.session_state.processed_frames = 0
        if "model" not in st.session_state:
            st.session_state.model = None

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title='Hands Detector', page_icon='✌',
                   layout='centered', initial_sidebar_state='expanded')
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)
        self.st.title("Ultralytics YOLO Streamlit应用")
        self.st.subheader("实时摄像头目标检测与跟踪")
        
    def update_fps(self):
        """Update the FPS value."""
        current_time = time.time()
        if self.prev_time != 0:
            self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_display.text(f"FPS: {self.fps:.2f}")  # Display FPS in sidebar
        
    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = "logo.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Video Source",
            ("webcam", "video"),
        )  # Add source selection dropdown
        self.enable_trk = self.st.sidebar.radio("启用跟踪", ("是", "否"), index=1)  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("置信度阈值", 0.0, 1.0, self.conf, 0.01)
        )  # Slider for confidence
        self.iou = float(self.st.sidebar.slider("IoU 阈值", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
        self.org_frame = col1.empty()  # Container for original frame
        self.ann_frame = col2.empty()  # Container for annotated frame
        self.fps_display = self.st.sidebar.empty()

    # 文件上传处理
    def source_upload(self):
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                # 使用tempfile生成唯一文件名
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(vid_file.read())
                    self.vid_file_name = tmp.name
        elif self.source == "webcam":
            # 添加摄像头索引选择
            camera_indices = self.getcamsearch()
            self.camsearch = self.st.sidebar.selectbox("选择摄像头", camera_indices)
            self.vid_file_name = self.camsearch
            pass

    def getcamsearch(self):
        # 自动检测可用摄像头索引
        indices = []
        for i in range(5):  # 尝试前5个索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                indices.append(i)
                cap.release()
            if len(indices) >= 2:  # 发现足够多的摄像头后停止
                break
        return indices

    def configure(self):
        """Configure the model and load selected classes for inference."""
        # Add dropdown menu for model selection
        available_models = [x.replace("yolo", "YOLO") for x in ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:  # If user provided the custom model, insert model without suffix as *.pt is added later
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")  # Load the YOLO model
            class_names = list(self.model.names.values())  # Convert dictionary to list of class names
        self.st.success("Model loaded successfully!")

        # Multiselect box with class names and get indices of selected classes
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, List):  # Ensure selected_options is a list
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui() # Initialize the web interface 
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        self.configure()  # Configure the app

        if self.st.sidebar.button("Start"):
            stop_button_pressed = False

            def stop_inference():
                nonlocal stop_button_pressed
                stop_button_pressed = True

            stop_button = self.st.button("Stop", on_click=stop_inference)  # Button to stop the inference
            cap = cv2.VideoCapture(self.vid_file_name)  # Capture the video
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return

            try:
                while cap.isOpened() and not stop_button_pressed:
                    success, frame = cap.read()
                    if not success:
                        self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                        break
                    self.update_fps()
                    
                    # Process frame with model
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind,
                            tracker="bytetrack.yaml", persist=True, stream=True
                        )
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, imgsz=640)

                    annotated_frame = results[0].plot()  # Add annotations on frame

                    self.org_frame.image(frame, channels="BGR")  # Display original frame
                    self.ann_frame.image(annotated_frame, channels="BGR")  # Display processed frame

            except Exception as e:
                self.st.error(f"An error occurred: {e}")
            finally:
                cap.release()  # Release the capture
                cv2.destroyAllWindows()  # Destroy all OpenCV windows
                if self.vid_file_name != 0 and os.path.exists(self.vid_file_name):
                    os.remove(self.vid_file_name)  # Clean up the temporary video file


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()