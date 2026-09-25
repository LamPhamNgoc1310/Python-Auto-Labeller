import cv2
import torch
import queue
import threading
import multiprocessing
import numpy as np
import re  # Added for IP parsing
from ultralytics import YOLO
import logging

# Ensure this matches your actual filename
from test_model_gpu_decoder import GPUVideoDecoder

# --- CONFIG ---
MODEL_ENGINE_PATH = r"C:\Khang\ai-nvdec-snapshot-2task-processor-a\ai-nvdec-snapshot-2task-processor\models\ModelHN_240326_03_Yolo26s_150526_01.engine"
THUMB_W, THUMB_H = 640, 480 

INFERENCE_CONFIG = {
    "conf": 0.4,
    "iou": 0.45,
    "imgsz": [480, 640],
    "classes": [0],
    "verbose": False
}

class CamThread(threading.Thread):
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        
        # --- NEW: Extract the last part of the IP (e.g., .197) ---
        match = re.search(r'192\.168\.1\.(\d+)', url)
        self.cam_id = match.group(1) if match else "???"

    def run(self):
        cap = GPUVideoDecoder(self.url, width=THUMB_W, height=THUMB_H)
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            if self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except queue.Empty: pass
            self.frame_queue.put(frame)
        
        cap.release()

def gpu_worker_process(gpu_id, camera_urls):
    model = YOLO(MODEL_ENGINE_PATH, task="detect")
    
    handlers = []
    for url in camera_urls:
        h = CamThread(url)
        h.start()
        handlers.append(h)

    latest_annotated = [None] * len(handlers)

    while True:
        frames_to_batch = []
        active_indices = []

        for i, h in enumerate(handlers):
            try:
                frames_to_batch.append(h.frame_queue.get_nowait())
                active_indices.append(i)
            except queue.Empty:
                continue

        if frames_to_batch:
            results = model.predict(
                frames_to_batch, 
                device=gpu_id, 
                **INFERENCE_CONFIG
            )

            for i, res in enumerate(results):
                idx = active_indices[i]
                annotated = res.plot()
                
                # --- NEW: Label with the extracted IP ID ---
                label = f"CAM: .{handlers[idx].cam_id}"
                cv2.putText(annotated, label, (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                latest_annotated[idx] = annotated

        # Display Logic
        valid_frames = [f for f in latest_annotated if f is not None]
        if valid_frames:
            rows = []
            for i in range(0, len(valid_frames), 3):
                row_chunk = valid_frames[i:i+3]
                row = np.hstack(row_chunk)
                if row.shape[1] < (THUMB_W * 3):
                    padding = np.zeros((THUMB_H, (THUMB_W * 3) - row.shape[1], 3), dtype=np.uint8)
                    row = np.hstack([row, padding])
                rows.append(row)
            
            if rows:
                final_grid = np.vstack(rows)
                # Resize for easier viewing on standard monitors
                display_h = 900
                display_w = int(final_grid.shape[1] * (display_h / final_grid.shape[0]))
                cv2.imshow(f"GPU {gpu_id} View", cv2.resize(final_grid, (display_w, display_h)))

        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    ALL_SOURCES = [
        # "rtsp://admin:Thado12@@192.168.1.144:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.145:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.146:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.149:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.150:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.151:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.162:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.164:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.167:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.169:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.159:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.165:554/Streaming/Channels/102",

        # "rtsp://admin:Thado12@@192.168.1.180:554/Streaming/Channels/102",
         "rtsp://admin:Thado12@@192.166.1.174:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.184:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.193:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.192:554/Streaming/Channels/102",
        #"rtsp://admin:Thado12@@192.168.1.174:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.178:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.179:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.172:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.175:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.170:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.143:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.130:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.116:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.114:554/Streaming/Channels/102",
        # "rtsp://admin:Thado12@@192.168.1.112:554/Streaming/Channels/102",
    ]
    
    mid = len(ALL_SOURCES) // 2
    p1 = multiprocessing.Process(target=gpu_worker_process, args=(0, ALL_SOURCES[:mid]))
    p2 = multiprocessing.Process(target=gpu_worker_process, args=(1, ALL_SOURCES[mid:]))

    p1.start()
    p2.start()
    p1.join()
    p2.join()