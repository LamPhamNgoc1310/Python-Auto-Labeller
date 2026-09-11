import pathlib
import torch
from ultralytics import YOLO

# 1. THE FIX: Patch PosixPath to WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

def export_model():
    # 2. Load your model
    model_path = r"E:\lampn\Programming\iRaypleMVT-KortekDetection\Models\detect_20260703160909_best.smartdlxmodel"
    model = YOLO(model_path)
    
    # 3. Run the export
    # This will now work on your RTX 3060 and TensorRT 10.15
    model.export(format='engine', device=-1, half=True, dynamic=True, batch=32, imgsz=[480, 640])

if __name__ == "__main__":
    export_model()