import pathlib
import torch
from ultralytics import YOLO

# 1. THE FIX: Patch PosixPath to WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

def export_model():
    # 2. Load your model
    model_path = r"E:\lampn\Programming\MechPartsDetection\AutoLabeller\model\2026-09-14\run-2\weights\KortexFLYolo26s14092602.pt"
    model = YOLO(model_path)
    
    # 3. Run the export
    # This will now work on your RTX 3060 and TensorRT 10.15
    model.export(format='engine', device=0, half=True, dynamic=True, batch=4, imgsz=[720, 1280])

if __name__ == "__main__":
    export_model()