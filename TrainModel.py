import os
import random
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def run_training_process(model_pt, task_data, params, split_ratio=0.8):
    """
    task_data is a list of tuples: [(path, weight), (path, weight), ...]
    """
    # Fix for Windows Multi-GPU libuv error
    os.environ["USE_LIBUV"] = "0" 
    
    try:
        if not task_data:
            raise ValueError("Task list is empty.")

        model = YOLO(model_pt, task='detect')
        primary_path = Path(task_data[0][0])
        
        # 1. Structure the /model directory at project root
        model_root = Path(params.pop("model_dir", Path(__file__).parent / "model"))
        model_root.mkdir(parents=True, exist_ok=True)
        
        # 2. Date-based session folder (YYYY-MM-DD)
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = model_root / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Auto-increment run numbering (run-1, run-2, ...)
        existing_runs = [d.name for d in date_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
        run_numbers = []
        for r in existing_runs:
            try:
                run_numbers.append(int(r.split("-")[1]))
            except (IndexError, ValueError):
                pass

        next_run_num = max(run_numbers, default=0) + 1
        run_name = f"run-{next_run_num}"
        run_dir = date_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[TRAINING] Target Session: {date_str} | Run: {run_name}")
        print(f"[TRAINING] Artifacts directory: {run_dir.resolve().as_posix()}")

        all_train_images = []
        all_val_images = []
        valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

        for folder_path, weight in task_data:
            folder_p = Path(folder_path)
            imgs = []

            img_dir = folder_p / "images"
            if img_dir.exists():
                for ext in valid_extensions:
                    imgs.extend(list(img_dir.glob(ext)))

            if not imgs:
                for ext in valid_extensions:
                    imgs.extend(list(folder_p.rglob(f"obj_train_data*/{ext}")))

            if not imgs:
                for ext in valid_extensions:
                    imgs.extend(list(folder_p.glob(ext)))

            abs_imgs = [img.resolve().as_posix() for img in imgs]
            if not abs_imgs:
                continue

            random.shuffle(abs_imgs)
            
            if len(abs_imgs) == 1:
                train_part = abs_imgs
                val_part = abs_imgs
            else:
                split_idx = int(len(abs_imgs) * split_ratio)
                split_idx = max(1, min(split_idx, len(abs_imgs) - 1))
                train_part = abs_imgs[:split_idx]
                val_part = abs_imgs[split_idx:]
            
            all_train_images.extend(train_part * weight)
            all_val_images.extend(val_part)

        if not all_train_images:
            raise ValueError("No images found in the selected tasks.")

        random.shuffle(all_train_images)

        # Store manifests directly in the run directory
        train_txt = run_dir / "train_manifest.txt"
        val_txt = run_dir / "val_manifest.txt"
        
        with open(train_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_train_images))
        with open(val_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_val_images))

        dataset_names = None
        yaml_source = primary_path / "data.yaml"
        if yaml_source.exists():
            try:
                with open(yaml_source, 'r', encoding='utf-8') as yf:
                    loaded_yaml = yaml.safe_load(yf)
                    if loaded_yaml and 'names' in loaded_yaml:
                        dataset_names = loaded_yaml['names']
            except Exception:
                pass
        
        if not dataset_names:
            dataset_names = model.names

        data_yaml = {
            'train': train_txt.resolve().as_posix(),
            'val': val_txt.resolve().as_posix(),
            'nc': len(dataset_names),
            'names': dataset_names
        }
        
        yaml_path = run_dir / "data_config.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, sort_keys=False)

        params.pop('project', None)
        params.pop('name', None)

        # Launch YOLO training directly into date_dir / run_name
        model.train(
            data=str(yaml_path.resolve()), 
            project=str(date_dir.resolve()), 
            name=run_name, 
            exist_ok=True, 
            **params
        )

    except Exception as e:
        print(f"PROCESS ERROR: {e}")