import os
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

def run_training_process(model_pt, task_data, params, split_ratio=0.8):
    """
    task_data is now a list of tuples: [(path, weight), (path, weight), ...]
    """
    # Fix for Windows Multi-GPU libuv error
    os.environ["USE_LIBUV"] = "0" 
    
    try:
        model = YOLO(model_pt, task='detect')
        
        # FIX: task_data[0] is a tuple (path, weight). We only want the path [0].
        primary_path = Path(task_data[0][0]) 
        
        all_train_images = []
        all_val_images = []

        # Loop through each task and its specific weight
        for folder_path, weight in task_data:
            # folder_path is the string path, weight is the integer multiplier
            folder_p = Path(folder_path)
            imgs = list(folder_p.rglob("obj_train_data*/*.jpg"))
            abs_imgs = [str(img.absolute()) for img in imgs]
            
            if not abs_imgs:
                continue

            random.shuffle(abs_imgs)
            split_idx = int(len(abs_imgs) * split_ratio)
            
            train_part = abs_imgs[:split_idx]
            val_part = abs_imgs[split_idx:]
            
            # OVERSAMPLING: Multiply ONLY the training images by the weight
            all_train_images.extend(train_part * weight)
            
            # Validation images remain unique (no multiplication)
            all_val_images.extend(val_part)

        if not all_train_images:
            raise ValueError("No images found in the selected tasks.")

        # Shuffle the final weighted training list
        random.shuffle(all_train_images)

        # Create manifests in the first task's directory
        train_txt = primary_path / "train_manifest.txt"
        val_txt = primary_path / "val_manifest.txt"
        
        with open(train_txt, 'w') as f: f.write("\n".join(all_train_images))
        with open(val_txt, 'w') as f: f.write("\n".join(all_val_images))

        # Update/Create YAML
        data_yaml = {
            'train': str(train_txt.absolute()),
            'val': str(val_txt.absolute()),
            'nc': len(model.names),
            'names': model.names
        }
        yaml_path = primary_path / "data_config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)

        # Start Training
        model.train(data=str(yaml_path), project=str(primary_path), **params)

    except Exception as e:
        print(f"PROCESS ERROR: {e}")