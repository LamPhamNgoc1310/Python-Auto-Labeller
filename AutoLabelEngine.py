import cv2
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import math
from datetime import datetime

def run_auto_label_process(model_path, video_files, export_root, frame_interval=5, conf_threshold=0.45, log_callback=None):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(export_root) / f"auto_{ts}"
        data_dir = session_dir / "obj_train_data"
        os.makedirs(data_dir, exist_ok=True)

        if log_callback: log_callback(f"Session: {session_dir.name}")

        for idx, video in enumerate(video_files):
            v_path = Path(video)
            cap = cv2.VideoCapture(str(v_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Dynamically calculate gap based on user input (frame_interval in seconds)
            gap = int(fps * frame_interval)
            
            count, saved = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if count % gap == 0:
                    cv2.imwrite(str(data_dir / f"v{idx}_{v_path.stem}_f{saved}.jpg"), frame)
                    saved += 1
                count += 1
            cap.release()
            if log_callback: log_callback(f"Extracted {saved} frames from {v_path.name} (Interval: {frame_interval}s)")

        model = YOLO(model_path)
        model.predict(source=str(data_dir), conf=conf_threshold, save_txt=True, project=str(session_dir), name="p")

        label_src = session_dir / "p" / "labels"
        if label_src.exists():
            for f in label_src.glob("*.txt"):
                shutil.move(str(f), str(data_dir / f.name))
        if (session_dir / "p").exists(): shutil.rmtree(session_dir / "p")

        return session_dir
    except Exception as e:
        if log_callback: log_callback(f"Labeler Error: {e}")
        raise e


def split_session_into_parts(session_dir, num_parts, log_callback=None):
    try:
        data_dir = Path(session_dir) / "obj_train_data"
        images = sorted(list(data_dir.glob("*.jpg")))
        total_imgs = len(images)
        
        if total_imgs == 0 or num_parts <= 1:
            return [] # Return empty list if no split happened

        imgs_per_part = math.ceil(total_imgs / num_parts)
        part_paths = [] # Track the new folders

        for i in range(num_parts):
            part_name = f"Part_{i+1}_{session_dir.name}"
            part_dir = session_dir / f"part_{i+1}"
            part_data_dir = part_dir / "obj_train_data"
            os.makedirs(part_data_dir, exist_ok=True)
            
            start_idx = i * imgs_per_part
            end_idx = min(start_idx + imgs_per_part, total_imgs)
            
            for img_path in images[start_idx:end_idx]:
                shutil.move(str(img_path), str(part_data_dir / img_path.name))
                lbl_path = img_path.with_suffix('.txt')
                if lbl_path.exists():
                    shutil.move(str(lbl_path), str(part_data_dir / lbl_path.name))
            
            part_paths.append((part_name, part_dir))
        
        if not any(data_dir.iterdir()):
            shutil.rmtree(data_dir)
            
        return part_paths # Return the list of (name, path) tuples
    except Exception as e:
        if log_callback: log_callback(f"Split Error: {e}")
        return []