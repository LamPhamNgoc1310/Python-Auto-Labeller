import cv2
import os
import time
import shutil
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext, simpledialog
from pathlib import Path
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import threading
from multiprocessing import Process
import json
import subprocess
import sys

# Import custom modules
import AutoLabelEngine
import TrainModel

# Fix for potential path issues
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

class LabelMakerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Active Learning Suite - Multi-Engine Edition")
        self.root.geometry("1500x950")

        # --- Variables ---
        self.model_path = tk.StringVar(value="yolov8s.pt")
        self.autolabel_model_path = tk.StringVar(value="")
        self.export_dir = tk.StringVar(value=str(Path(__file__).parent / "export"))
        self.model_dir = tk.StringVar(value=str(Path(__file__).parent / "model"))
        self.video_files = []
        
        # Ensure base directories exist
        os.makedirs(self.export_dir.get(), exist_ok=True)
        os.makedirs(self.model_dir.get(), exist_ok=True)
        
        self.db_path = Path(__file__).parent / "tasks_db.json"
        self.tasks = self.load_tasks_db()
        self.oversample_factor = tk.StringVar(value="1")

        self.training_session_path = None
        self.image_list = []
        self.current_img_idx = 0

        self.class_names = {0: "object"} 
        self.current_class_id = tk.IntVar(value=0)
        self.class_colors = [
            "#2ecc71", "#e74c3c", "#3498db", "#f1c40f", 
            "#9b59b6", "#e67e22", "#1abc9c", "#e84393", "#34495e"
        ]

        self.undo_stack = []
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.is_dirty = False
        
        # --- Editor State ---
        self.boxes = [] # [cls, x1, y1, x2, y2]
        self.selected_box_idx = -1
        self.drawing_box = None
        self.resizing = False
        self.train_proc = None

        # --- Hyperparameters ---
        self.train_params = {
            "epochs": 100,
            "batch": 16,
            "patience": 30,
            "workers": 2,
            "cos_lr": "true",
            "lr0": 0.01,
            "lrf": 0.01,
            "imgsz": 640,
            "device": "-1",
            "box": 10.0,
            "dfl": 2.5,
            "close_mosaic": 20,
            "freeze": 0,
            "mosaic": 1.0,
            "mixup": 0.8,
            "fliplr": 0.5,
            "scale": 0.5,
            "translate": 0.1,
            "degrees": 12,
            "perspective": 0.001,
            "shear" : 2.0,
            "hsv_h": 0.015,
            "hsv_s": 0.25,
            "hsv_v": 0.3,
        }
        self.param_vars = {k: tk.StringVar(value=v) for k, v in self.train_params.items()}
        self.train_split_ratio = tk.DoubleVar(value=0.85)
        self.extraction_interval = tk.StringVar(value="5")
        self.num_split_parts = tk.StringVar(value="1")

        self.create_widgets()
        self.refresh_task_list()

    def load_tasks_db(self):
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return []

    def save_task_to_db(self, name, path, task_type="import", classes=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        if classes is None:
            classes = {0: "object"}
            
        new_task = {
            "name": name,
            "path": str(path),
            "date": now,
            "type": task_type,
            "edited": now,
            "classes": {str(k): v for k, v in classes.items()}
        }
        
        self.tasks.append(new_task)
        self.save_json()

    def write_data_yaml(self, task_dir, classes_dict):
        yaml_path = Path(task_dir) / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write("names:\n")
            for c_id in sorted([int(k) for k in classes_dict.keys()]):
                f.write(f"  {c_id}: {classes_dict[c_id]}\n")

    # ==========================================
    # UI ARCHITECTURE
    # ==========================================
    def create_widgets(self):
        header = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="YOLO ACTIVE LEARNING SUITE", fg="white", 
                 bg="#2c3e50", font=("Arial", 12, "bold")).pack()

        paned = tk.PanedWindow(self.root, orient="horizontal", 
                               sashwidth=12, sashpad=2, 
                               bg="#7f8c8d", showhandle=True)
        paned.pack(fill="both", expand=True)

        # ------------------------------------------
        # LEFT PANEL: DATA, TASKS, IMPORT/EXPORT & NAV
        # ------------------------------------------
        left_panel = tk.Frame(paned, padx=5, pady=5)
        paned.add(left_panel, width=390)

        task_frame = tk.LabelFrame(left_panel, text="Task Database (Ctrl+Click to select multiple)", padx=5, pady=5)
        task_frame.pack(fill="both", expand=True, pady=(0, 5))

        cols = ("idx", "name", "weight", "created", "edited")
        self.task_tree = ttk.Treeview(task_frame, columns=cols, show="headings", selectmode="extended")
        
        self.task_tree.heading("idx", text="#")
        self.task_tree.heading("name", text="Name")
        self.task_tree.heading("weight", text="W(x)")
        self.task_tree.heading("created", text="Added")
        self.task_tree.heading("edited", text="Edited")
        
        self.task_tree.column("idx", width=30, anchor="center")
        self.task_tree.column("name", width=120)
        self.task_tree.column("weight", width=45, anchor="center")
        self.task_tree.column("created", width=80)
        self.task_tree.column("edited", width=80)
        
        self.task_tree.pack(fill="both", expand=True)
        self.task_tree.bind("<<TreeviewSelect>>", self.on_task_select)

        sel_frame = tk.Frame(task_frame)
        sel_frame.pack(fill="x", pady=2)
        tk.Button(sel_frame, text="☑ Select All", command=self.select_all_tasks).pack(side="left", expand=True, fill="x", padx=(0,2))
        tk.Button(sel_frame, text="☒ Clear", command=self.clear_task_selection).pack(side="left", expand=True, fill="x", padx=(2,0))

        data_frame = tk.LabelFrame(left_panel, text="YOLO Import & Export", padx=5, pady=5)
        data_frame.pack(fill="x", pady=5)
        tk.Button(data_frame, text="📦 Bulk Import Standard YOLO (.zip)", command=self.import_zip, bg="#9b59b6", fg="white").pack(fill="x", pady=2)
        tk.Button(data_frame, text="📤 Bulk Export Selected Tasks (.zip)", bg="#2980b9", fg="white", command=self.export_task).pack(fill="x", pady=2)

        action_frame = tk.LabelFrame(left_panel, text="Task Actions", padx=5, pady=5)
        action_frame.pack(fill="x", pady=5)
        
        btn_row = tk.Frame(action_frame)
        btn_row.pack(fill="x", pady=2)
        tk.Button(btn_row, text="⚖ Weight", bg="#16a085", fg="white", command=self.set_task_weight).pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(btn_row, text="✏ Rename", command=self.rename_task).pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Button(action_frame, text="🗑 Delete Task", bg="#c0392b", fg="white", command=self.delete_task).pack(fill="x", pady=2)

        nav_frame = tk.LabelFrame(left_panel, text="Editor Navigation", padx=5, pady=5, bg="#34495e", fg="white")
        nav_frame.pack(fill="x", pady=5)
        
        nav_top = tk.Frame(nav_frame, bg="#34495e")
        nav_top.pack(fill="x", pady=2)
        tk.Button(nav_top, text="◀ PREV", command=self.prev_img, width=8).pack(side="left", padx=5)
        self.idx_lbl = tk.Label(nav_top, text="0/0", fg="white", bg="#34495e", font=("Arial", 10, "bold"))
        self.idx_lbl.pack(side="left", expand=True)
        tk.Button(nav_top, text="NEXT ▶", command=self.next_img, width=8).pack(side="right", padx=5)

        img_mod_frame = tk.Frame(nav_frame, bg="#34495e")
        img_mod_frame.pack(fill="x", pady=3)
        tk.Button(img_mod_frame, text="➕ Add Image(s)", bg="#27ae60", fg="white", font=("Arial", 8, "bold"), 
                  command=self.add_images_to_task).pack(side="left", fill="x", expand=True, padx=(2, 2))
        tk.Button(img_mod_frame, text="🗑 Delete Image", bg="#c0392b", fg="white", font=("Arial", 8, "bold"), 
                  command=self.delete_current_image).pack(side="left", fill="x", expand=True, padx=(2, 2))

        tk.Button(nav_frame, text="⚡ SPOT-LABEL CURRENT IMAGE", bg="#3498db", fg="white", command=self.autolabel_current_view).pack(fill="x", padx=5, pady=5)
        tk.Label(nav_frame, text="Auto-Save Enabled", fg="#2ecc71", bg="#34495e", font=("Arial", 8, "italic")).pack(pady=2)

        # ------------------------------------------
        # MIDDLE PANEL: PIPELINES
        # ------------------------------------------
        mid_frame = tk.Frame(paned, padx=10, pady=10)
        paned.add(mid_frame, width=410)

        self.ctrl_tabs = ttk.Notebook(mid_frame)
        self.ctrl_tabs.pack(fill="both", expand=True)

        self.pipe_tab = tk.Frame(self.ctrl_tabs, padx=5, pady=5)
        self.hyper_tab = tk.Frame(self.ctrl_tabs, padx=5, pady=5)
        self.ctrl_tabs.add(self.pipe_tab, text="Pipelines")
        self.ctrl_tabs.add(self.hyper_tab, text="Hyperparams")

        self.build_pipeline_ui()
        self.build_scrollable_hyper_tab()

        # ------------------------------------------
        # RIGHT PANEL: CANVAS EDITOR
        # ------------------------------------------
        right_frame = tk.Frame(paned, bg="#34495e")
        paned.add(right_frame)

        cls_f = tk.Frame(right_frame, bg="#34495e", pady=5)
        cls_f.pack(fill="x")
        tk.Label(cls_f, text="Active Class:", fg="white", bg="#34495e").pack(side="left", padx=10)
        
        self.cls_dropdown = ttk.Combobox(cls_f, state="readonly", width=15)
        self.cls_dropdown.pack(side="left", padx=5)
        self.cls_dropdown.bind("<<ComboboxSelected>>", self.update_active_class)
        
        tk.Button(cls_f, text="+ Add", bg="#2ecc71", fg="white", font=("Arial", 8, "bold"), 
                  command=self.add_class).pack(side="left", padx=5)
        tk.Button(cls_f, text="- Del", bg="#e74c3c", fg="white", font=("Arial", 8, "bold"), 
                  command=self.delete_class).pack(side="left", padx=5)
        
        self.refresh_class_dropdown()

        self.editor_canvas = tk.Canvas(right_frame, bg="#1e1e1e", cursor="cross")
        self.editor_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.editor_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.editor_canvas.bind("<B1-Motion>", self.on_move_press)
        self.editor_canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.editor_canvas.bind("<Button-3>", self.on_right_click)

    def select_all_tasks(self):
        for item in self.task_tree.get_children():
            self.task_tree.selection_add(item)

    def clear_task_selection(self):
        self.task_tree.selection_remove(self.task_tree.get_children())
        self.training_session_path = None
        self.image_list = []
        self.boxes = []
        self.editor_canvas.delete("all")
        self.idx_lbl.config(text="0/0")

    def build_pipeline_ui(self):
        # Pipeline A1: Video Frame Extraction
        ext_box = tk.LabelFrame(self.pipe_tab, text="Pipeline A1: Video Frame Extraction (No Model)", pady=8, padx=8)
        ext_box.pack(fill="x", pady=4)

        tk.Label(ext_box, text="Video Queue:").pack(anchor="w")
        v_list_frame = tk.Frame(ext_box)
        v_list_frame.pack(fill="x", pady=2)

        self.v_listbox = tk.Listbox(v_list_frame, height=4, font=("Arial", 8))
        self.v_listbox.pack(side="left", fill="x", expand=True)
        v_scroll = tk.Scrollbar(v_list_frame, orient="vertical", command=self.v_listbox.yview)
        v_scroll.pack(side="right", fill="y")
        self.v_listbox.config(yscrollcommand=v_scroll.set)

        v_btn_f = tk.Frame(ext_box)
        v_btn_f.pack(fill="x", pady=2)
        tk.Button(v_btn_f, text="+ Add Video(s)", command=self.browse_videos).pack(side="left", expand=True, fill="x")
        tk.Button(v_btn_f, text="- Remove Selected", command=self.remove_selected_video).pack(side="left", expand=True, fill="x")

        f_opts = tk.Frame(ext_box)
        f_opts.pack(fill="x", pady=2)
        tk.Label(f_opts, text="Every (sec):").pack(side="left")
        tk.Entry(f_opts, textvariable=self.extraction_interval, width=6).pack(side="left", padx=5)
        tk.Label(f_opts, text="Split into parts:").pack(side="left", padx=(10, 0))
        tk.Entry(f_opts, textvariable=self.num_split_parts, width=4).pack(side="left", padx=5)

        self.extract_btn = tk.Button(ext_box, text="🎞 EXTRACT FRAMES TO NEW TASK", bg="#16a085", fg="white", font=("Arial", 9, "bold"), command=self.start_frame_extraction)
        self.extract_btn.pack(fill="x", pady=4)

        # Pipeline A2: Model Auto-Labeling
        auto_box = tk.LabelFrame(self.pipe_tab, text="Pipeline A2: Batch Auto-Labeling (Target Selected Task)", pady=8, padx=8)
        auto_box.pack(fill="x", pady=4)

        tk.Label(auto_box, text="Inference Model (.pt):").pack(anchor="w")
        m_frame = tk.Frame(auto_box)
        m_frame.pack(fill="x", pady=2)
        tk.Entry(m_frame, textvariable=self.autolabel_model_path).pack(side="left", fill="x", expand=True)
        tk.Button(m_frame, text="...", command=self.browse_autolabel_model).pack(side="right")

        self.autolabel_task_btn = tk.Button(auto_box, text="⚡ AUTO-LABEL SELECTED TASK", bg="#3498db", fg="white", font=("Arial", 9, "bold"), command=self.start_batch_autolabel)
        self.autolabel_task_btn.pack(fill="x", pady=4)

        # Pipeline B: Model Training
        train_box = tk.LabelFrame(self.pipe_tab, text="Pipeline B: Model Training", pady=8, padx=8)
        train_box.pack(fill="x", pady=4)
        
        tk.Label(train_box, text="BASE MODEL TO TRAIN:").pack(anchor="w")
        m_frame2 = tk.Frame(train_box)
        m_frame2.pack(fill="x", pady=2)
        tk.Entry(m_frame2, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        tk.Button(m_frame2, text="...", command=self.browse_model).pack(side="right", padx=2)

        self.train_btn = tk.Button(train_box, text="🔥 START TRAINING", bg="#e67e22", fg="white", height=2, command=self.start_training)
        self.train_btn.pack(fill="x", pady=10)
        
        tk.Button(train_box, text="🛑 CANCEL TRAINING", bg="#c0392b", fg="white", command=self.stop_training).pack(fill="x", pady=2)

        # Log Console
        tk.Label(self.pipe_tab, text="Activity Log:").pack(anchor="w", pady=(6, 0))
        self.log_area = scrolledtext.ScrolledText(self.pipe_tab, height=8, font=("Consolas", 8))
        self.log_area.pack(fill="both", expand=True)

    # ==========================================
    # PIPELINE IMPLEMENTATIONS
    # ==========================================
    def start_frame_extraction(self):
        if not self.video_files:
            return messagebox.showwarning("Warning", "Please add at least one video to the queue!")
        threading.Thread(target=self._exec_frame_extraction, daemon=True).start()

    def _exec_frame_extraction(self):
        self.extract_btn.config(state="disabled")
        try:
            interval = float(self.extraction_interval.get())
            parts_count = int(self.num_split_parts.get())
            if interval <= 0:
                interval = 1.0

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = Path(self.export_dir.get()) / f"extracted_{ts}"
            images_dir = session_dir / "images"
            labels_dir = session_dir / "labels"
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            total_frames = 0
            for v_path_str in self.video_files:
                v_path = Path(v_path_str)
                cap = cv2.VideoCapture(str(v_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_step = max(1, int(fps * interval))

                frame_idx = 0
                saved_count = 0
                self.log(f"Extracting frames from: {v_path.name} (every {interval}s)...")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % frame_step == 0:
                        img_name = f"{v_path.stem}_f{frame_idx:06d}.jpg"
                        cv2.imwrite(str(images_dir / img_name), frame)
                        (labels_dir / f"{v_path.stem}_f{frame_idx:06d}.txt").touch()
                        saved_count += 1
                        total_frames += 1
                    frame_idx += 1
                cap.release()
                self.log(f"Extracted {saved_count} frames from {v_path.name}")

            if total_frames == 0:
                self.log("Extraction finished with 0 frames.")
                shutil.rmtree(session_dir, ignore_errors=True)
                return

            self.write_data_yaml(session_dir, self.class_names)

            if parts_count > 1:
                all_imgs = sorted(list(images_dir.glob("*.jpg")))
                chunk_size = (len(all_imgs) + parts_count - 1) // parts_count
                
                for part_idx in range(parts_count):
                    part_chunk = all_imgs[part_idx * chunk_size : (part_idx + 1) * chunk_size]
                    if not part_chunk:
                        continue
                    part_dir = Path(self.export_dir.get()) / f"{session_dir.name}_part{part_idx + 1}"
                    p_img_dir = part_dir / "images"
                    p_lbl_dir = part_dir / "labels"
                    os.makedirs(p_img_dir, exist_ok=True)
                    os.makedirs(p_lbl_dir, exist_ok=True)

                    for img_f in part_chunk:
                        shutil.move(str(img_f), str(p_img_dir / img_f.name))
                        lbl_f = labels_dir / f"{img_f.stem}.txt"
                        if lbl_f.exists():
                            shutil.move(str(lbl_f), str(p_lbl_dir / lbl_f.name))
                        else:
                            (p_lbl_dir / f"{img_f.stem}.txt").touch()

                    self.write_data_yaml(part_dir, self.class_names)
                    self.save_task_to_db(part_dir.name, part_dir, "extracted_part", classes=self.class_names)

                shutil.rmtree(session_dir, ignore_errors=True)
                self.log(f"Frame extraction split into {parts_count} tasks.")
            else:
                self.save_task_to_db(session_dir.name, session_dir, "extracted", classes=self.class_names)
                self.training_session_path = session_dir
                self.load_previews()
                self.log(f"Frame extraction complete: {total_frames} frames registered.")

        except Exception as e:
            self.log(f"Extraction Error: {e}")
        finally:
            self.extract_btn.config(state="normal")
            self.refresh_task_list()

    def start_batch_autolabel(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("Warning", "Select a task from the Task Database to auto-label!")

        model_p = self.autolabel_model_path.get()
        if not model_p or not os.path.exists(model_p):
            return messagebox.showerror("Error", "Please select a valid Inference Model (.pt) first.")

        threading.Thread(target=self._exec_batch_autolabel, args=(selected_iids[0], model_p), daemon=True).start()

    def _exec_batch_autolabel(self, iid, model_path):
        self.autolabel_task_btn.config(state="disabled")
        try:
            actual_idx = int(self.task_tree.item(iid)['values'][0])
            task = self.tasks[actual_idx]
            task_path = Path(__file__).parent / task['path']
            images_dir = task_path / "images"
            labels_dir = task_path / "labels"
            os.makedirs(labels_dir, exist_ok=True)

            img_list = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                img_list.extend(list(images_dir.glob(ext)))

            if not img_list:
                self.log(f"No images found in {task['name']} to label.")
                return

            self.log(f"Loading model {Path(model_path).name} for batch auto-labeling...")
            model = YOLO(model_path)
            
            task_classes = {int(k): v for k, v in task.get("classes", {}).items()}
            updated_classes = False

            for i, img_p in enumerate(img_list):
                results = model.predict(source=str(img_p), conf=0.45, save=False, verbose=False)[0]
                label_txt = labels_dir / f"{img_p.stem}.txt"
                
                with open(label_txt, 'w') as f:
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        if cls not in task_classes:
                            task_classes[cls] = model.names.get(cls, f"class_{cls}")
                            updated_classes = True

                        xywhn = box.xywhn[0].tolist()
                        f.write(f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

                if (i + 1) % 10 == 0 or (i + 1) == len(img_list):
                    self.log(f"Auto-labeled {i + 1}/{len(img_list)} images in {task['name']}")

            self.tasks[actual_idx]["classes"] = {str(k): v for k, v in task_classes.items()}
            self.tasks[actual_idx]["edited"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.save_json()
            self.write_data_yaml(task_path, task_classes)

            if self.training_session_path and self.training_session_path.resolve() == task_path.resolve():
                self.class_names = task_classes
                self.refresh_class_dropdown()
                self.show_image()

            self.log(f"Batch auto-labeling completed for task: {task['name']}")
        except Exception as e:
            self.log(f"Batch Auto-Label Error: {e}")
        finally:
            self.autolabel_task_btn.config(state="normal")
            self.refresh_task_list()

    # ==========================================
    # BULK YOLO IMPORT & EXPORT
    # ==========================================
    def import_zip(self):
        paths = filedialog.askopenfilenames(filetypes=[("Zip files", "*.zip")], title="Select one or more YOLO Zips")
        if not paths: return
        
        imported_count = 0

        for p in paths:
            p = Path(p)
            ts = datetime.now().strftime("%H%M%S_%f")[:10]
            staging_dir = Path(self.export_dir.get()) / f"staging_{ts}"
            os.makedirs(staging_dir, exist_ok=True)
            
            self.log(f"Unpacking archive: {p.name}...")
            try:
                with zipfile.ZipFile(p, 'r') as z:
                    z.extractall(staging_dir)
                
                nested_zips = list(staging_dir.rglob("*.zip"))
                for nz in nested_zips:
                    try:
                        with zipfile.ZipFile(nz, 'r') as sub_z:
                            extract_sub = nz.parent / nz.stem
                            sub_z.extractall(extract_sub)
                        os.remove(nz)
                    except Exception as e:
                        self.log(f"Failed to unpack nested zip {nz.name}: {e}")

                found_yamls = list(staging_dir.rglob("data.yaml")) + list(staging_dir.rglob("dataset.yaml"))
                
                if not found_yamls:
                    self.log(f"No valid YOLO datasets (missing data.yaml) found in {p.name}")
                    shutil.rmtree(staging_dir, ignore_errors=True)
                    continue

                for yaml_path in found_yamls:
                    base_dir = yaml_path.parent
                    
                    if base_dir == staging_dir:
                        task_name = p.stem
                    else:
                        task_name = base_dir.name

                    task_classes = {}
                    with open(yaml_path, 'r') as f:
                        lines = f.readlines()
                        in_names = False
                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith("names:"):
                                in_names = True
                                continue
                            if in_names:
                                if not line.startswith(" ") and not line.startswith("-") and stripped.endswith(":"):
                                    in_names = False
                                    continue
                                if stripped.startswith("-"):
                                    task_classes[len(task_classes)] = stripped.replace("-", "", 1).strip().strip("\"'")
                                elif ":" in stripped:
                                    parts = stripped.split(":", 1)
                                    if parts[0].strip().isdigit():
                                        task_classes[int(parts[0].strip())] = parts[1].strip().strip("\"'")

                    if not task_classes:
                        task_classes = {0: "object"}

                    src_img_dir = base_dir / "images"
                    src_lbl_dir = base_dir / "labels"
                    
                    if not src_img_dir.exists():
                        loose_images = []
                        for ext in ["*.jpg", "*.jpeg", "*.png"]:
                            loose_images.extend(list(base_dir.glob(ext)))
                        if not loose_images:
                            self.log(f"Skipping {task_name}: No images found.")
                            continue
                        
                        os.makedirs(src_img_dir, exist_ok=True)
                        for img in loose_images:
                            shutil.move(str(img), str(src_img_dir / img.name))

                    final_ts = datetime.now().strftime("%H%M%S_%f")[:10]
                    final_task_dir = Path(self.export_dir.get()) / f"yolo_{final_ts}"
                    os.makedirs(final_task_dir, exist_ok=True)
                    
                    target_img_dir = final_task_dir / "images"
                    target_lbl_dir = final_task_dir / "labels"
                    os.makedirs(target_img_dir, exist_ok=True)
                    os.makedirs(target_lbl_dir, exist_ok=True)
                    
                    for item in src_img_dir.glob("*"):
                        shutil.move(str(item), str(target_img_dir / item.name))
                        
                    if src_lbl_dir.exists():
                        for item in src_lbl_dir.glob("*"):
                            shutil.move(str(item), str(target_lbl_dir / item.name))

                    self.write_data_yaml(final_task_dir, task_classes)

                    try: rel_path = final_task_dir.relative_to(Path(__file__).parent)
                    except ValueError: rel_path = final_task_dir

                    self.save_task_to_db(task_name, rel_path, "import", classes=task_classes)
                    self.log(f"Imported task: {task_name} ({len(task_classes)} classes)")
                    imported_count += 1

            except Exception as e:
                self.log(f"YOLO Import Error on {p.name}: {e}")
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
        
        if imported_count > 0:
            messagebox.showinfo("Import Complete", f"Successfully extracted and imported {imported_count} task(s)!")
        self.refresh_task_list()

    def export_task(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("!", "Select one or more tasks to export.")
            
        if len(selected_iids) == 1:
            idx = int(self.task_tree.item(selected_iids[0])['values'][0])
            task = self.tasks[idx]
            task_path = Path(__file__).parent / task['path']
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".zip", 
                initialfile=f"{task['name']}_yolo.zip",
                filetypes=[("Zip files", "*.zip")]
            )
            if not file_path:
                return

            try:
                classes = {int(k): v for k, v in task.get("classes", {0:"object"}).items()}
                self.write_data_yaml(task_path, classes)
                
                base_name = str(file_path).replace('.zip', '')
                shutil.make_archive(base_name, 'zip', task_path)
                
                self.log(f"Exported YOLO task to: {file_path}")
                messagebox.showinfo("Export Complete", "Standard YOLO dataset package exported successfully!")
            except Exception as e:
                self.log(f"YOLO Export Failed: {e}")
                messagebox.showerror("Export Error", f"Failed to export: {e}")
                
        else:
            out_dir = filedialog.askdirectory(title=f"Select Output Folder for {len(selected_iids)} tasks")
            if not out_dir: 
                return
            
            out_dir = Path(out_dir)
            self.log(f"Starting bulk export of {len(selected_iids)} tasks...")
            success_count = 0
            
            for iid in selected_iids:
                try:
                    idx = int(self.task_tree.item(iid)['values'][0])
                    task = self.tasks[idx]
                    task_path = Path(__file__).parent / task['path']
                    
                    classes = {int(k): v for k, v in task.get("classes", {0:"object"}).items()}
                    self.write_data_yaml(task_path, classes)
                    
                    zip_base_path = out_dir / f"{task['name']}_yolo"
                    shutil.make_archive(str(zip_base_path), 'zip', task_path)
                    
                    success_count += 1
                    self.log(f"Exported: {task['name']}_yolo.zip")
                except Exception as e:
                    self.log(f"Failed to export '{task['name']}': {e}")
                    
            messagebox.showinfo("Bulk Export Complete", f"Successfully exported {success_count}/{len(selected_iids)} tasks to:\n{out_dir}")

    # ==========================================
    # DATASET IMAGE MANAGEMENT (ADD / DELETE)
    # ==========================================
    def add_images_to_task(self):
        if not self.training_session_path or not self.training_session_path.exists():
            return messagebox.showwarning("Warning", "Please select an active task to add images to.")

        file_paths = filedialog.askopenfilenames(
            title="Select Image(s) to Add to Dataset",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_paths:
            return

        images_dir = self.training_session_path / "images"
        labels_dir = self.training_session_path / "labels"
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        added_paths = []
        for src_str in file_paths:
            src_p = Path(src_str)
            dest_p = images_dir / src_p.name

            if dest_p.exists():
                ts = datetime.now().strftime("%H%M%S_%f")[:10]
                dest_p = images_dir / f"{src_p.stem}_{ts}{src_p.suffix}"

            shutil.copy(str(src_p), str(dest_p))

            lbl_p = labels_dir / f"{dest_p.stem}.txt"
            if not lbl_p.exists():
                lbl_p.touch()

            added_paths.append(dest_p)

        self.image_list.extend(added_paths)
        self.log(f"Added {len(added_paths)} image(s) to {self.training_session_path.name}.")

        selected = self.task_tree.selection()
        if selected:
            idx = int(self.task_tree.item(selected[0])['values'][0])
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.tasks[idx]['edited'] = now_str
            self.save_json()
            self.task_tree.set(selected[0], column="edited", value=now_str)

        self.update_stats()
        self.current_img_idx = len(self.image_list) - len(added_paths)
        self.show_image()

    def delete_current_image(self):
        if not self.image_list or not self.training_session_path:
            return messagebox.showwarning("Warning", "No image is currently loaded to delete.")

        img_p = self.image_list[self.current_img_idx]

        if messagebox.askyesno("Delete Image", f"Permanently delete '{img_p.name}' and its labels from disk?"):
            try:
                if img_p.exists():
                    os.remove(img_p)

                lbl_p = self.training_session_path / "labels" / f"{img_p.stem}.txt"
                if lbl_p.exists():
                    os.remove(lbl_p)

                sibling_txt = img_p.with_suffix('.txt')
                if sibling_txt.exists():
                    os.remove(sibling_txt)

                deleted_name = img_p.name
                self.image_list.pop(self.current_img_idx)
                self.log(f"Permanently deleted image from disk: {deleted_name}")

                selected = self.task_tree.selection()
                if selected:
                    idx = int(self.task_tree.item(selected[0])['values'][0])
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                    self.tasks[idx]['edited'] = now_str
                    self.save_json()
                    self.task_tree.set(selected[0], column="edited", value=now_str)

                if not self.image_list:
                    self.boxes = []
                    self.editor_canvas.delete("all")
                    self.idx_lbl.config(text="0/0")
                else:
                    if self.current_img_idx >= len(self.image_list):
                        self.current_img_idx = len(self.image_list) - 1
                    self.show_image()

                self.update_stats()

            except Exception as e:
                self.log(f"Error deleting image: {e}")
                messagebox.showerror("Error", f"Failed to delete image: {e}")

    # ==========================================
    # EDITOR & TASK ACTIONS
    # ==========================================
    def refresh_class_dropdown(self):
        values = [f"{k}: {v}" for k, v in self.class_names.items()]
        self.cls_dropdown['values'] = values
        
        if values:
            if self.current_class_id.get() in self.class_names:
                idx = list(self.class_names.keys()).index(self.current_class_id.get())
                self.cls_dropdown.current(idx)
            else:
                self.cls_dropdown.current(0)
                self.current_class_id.set(list(self.class_names.keys())[0])
        else:
            self.cls_dropdown.set('')
            self.current_class_id.set(-1)

    def add_class(self):
        new_name = simpledialog.askstring("Add Class", "Enter class name:")
        if new_name:
            new_id = max(self.class_names.keys()) + 1 if self.class_names else 0
            self.class_names[new_id] = new_name
            
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                self.save_json()
                if self.training_session_path:
                    self.write_data_yaml(self.training_session_path, self.class_names)
            
            self.current_class_id.set(new_id)
            self.refresh_class_dropdown()
            self.log(f"Class created: {new_id} -> {new_name}")

    def delete_class(self):
        if not self.class_names: return
        curr_id = self.current_class_id.get()
        curr_name = self.class_names.get(curr_id, "Unknown")
        
        if messagebox.askyesno("Delete Class", f"Delete class '{curr_id}: {curr_name}'?"):
            del self.class_names[curr_id]
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                self.save_json()
                if self.training_session_path:
                    self.write_data_yaml(self.training_session_path, self.class_names)
            
            original_len = len(self.boxes)
            self.boxes = [b for b in self.boxes if int(b[0]) != curr_id]
            if len(self.boxes) < original_len:
                self.is_dirty = True
                self.save_boxes()
                self.redraw()
            
            self.refresh_class_dropdown()
            self.log(f"Class deleted: {curr_name}")

    def refresh_task_list(self):
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
            
        for i, task in enumerate(self.tasks):
            t_weight = task.get('weight', 1) 
            self.task_tree.insert("", "end", values=(
                i, 
                task['name'], 
                f"{t_weight}x", 
                task.get('date', "N/A"), 
                task.get('edited', "N/A")
            ))

    def set_task_weight(self):
        selected = self.task_tree.selection()
        if not selected: return
        new_w = tk.simpledialog.askinteger("Weight", "Enter multiplier (1-50):", initialvalue=1, minvalue=1, maxvalue=50)
        if new_w:
            for iid in selected:
                idx = int(self.task_tree.item(iid)['values'][0])
                self.tasks[idx]['weight'] = new_w
                self.tasks[idx]['edited'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.save_json()
            self.refresh_task_list()

    def on_task_select(self, event):
        selected = self.task_tree.selection()
        if not selected: return
        
        idx = int(self.task_tree.item(selected[0])['values'][0])
        task = self.tasks[idx]
        self.training_session_path = Path(__file__).parent / task['path']
        
        if "classes" in task:
            self.class_names = {int(k): v for k, v in task["classes"].items()}
        else:
            self.class_names = {0: "object"}
            
        self.refresh_class_dropdown()
        self.update_stats()
        self.load_previews()

    def rename_task(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids: return
        
        idx = int(self.task_tree.item(selected_iids[0])['values'][0])
        old_name = self.tasks[idx]['name']
        new_name = tk.simpledialog.askstring("Rename", "New task name:", initialvalue=old_name)
        
        if new_name:
            self.tasks[idx]['name'] = new_name
            self.tasks[idx]['edited'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.save_json() 
            self.refresh_task_list()

    def delete_task(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("!", "Please select a task to delete.")

        if messagebox.askyesno("Confirm Delete", f"Delete {len(selected_iids)} task(s) from disk?"):
            indices = [int(self.task_tree.item(iid)['values'][0]) for iid in selected_iids]
            indices.sort(reverse=True)
            for idx in indices:
                if 0 <= idx < len(self.tasks):
                    try:
                        task_path = Path(__file__).parent / self.tasks[idx]['path']
                        if task_path.exists():
                            shutil.rmtree(task_path, ignore_errors=True)
                    except Exception as e:
                        self.log(f"Disk Delete Error: {e}")
                    self.tasks.pop(idx)
            
            self.save_json()
            self.refresh_task_list()
            
            if self.training_session_path and not self.training_session_path.exists():
                self.training_session_path = None
                self.image_list = []
                self.boxes = []
                self.editor_canvas.delete("all")
                self.idx_lbl.config(text="0/0")

    def remove_selected_video(self):
        selection = self.v_listbox.curselection()
        if not selection: return
        for index in reversed(selection):
            self.video_files.pop(index)
            self.v_listbox.delete(index)

    def build_scrollable_hyper_tab(self):
        self.hyper_canvas = tk.Canvas(self.hyper_tab)
        scrollbar = ttk.Scrollbar(self.hyper_tab, orient="vertical", command=self.hyper_canvas.yview)
        self.scrollable_frame = tk.Frame(self.hyper_canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.hyper_canvas.configure(scrollregion=self.hyper_canvas.bbox("all")))
        self.hyper_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.hyper_canvas.configure(yscrollcommand=scrollbar.set)
        self.hyper_tab.bind_all("<MouseWheel>", lambda e: self.hyper_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        stats_frame = tk.LabelFrame(self.scrollable_frame, text="Dataset Distribution", padx=10, pady=10)
        stats_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(stats_frame, text="Train/Val Split Ratio:").pack(side="left")
        split_slider = tk.Scale(stats_frame, from_=0.5, to_=0.95, resolution=0.05, 
                                orient="horizontal", variable=self.train_split_ratio, command=self.update_stats)
        split_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.stats_lbl = tk.Label(stats_frame, text="Total: 0 | Train: 0 | Val: 0", fg="#3498db", font=("Arial", 9, "bold"))
        self.stats_lbl.pack(fill="x", pady=5)

        tk.Label(self.scrollable_frame, text="YOLO Hyperparameters", font=("Arial", 10, "bold")).pack(pady=10)
        grid_f = tk.Frame(self.scrollable_frame)
        grid_f.pack(padx=20)
        for i, (name, var) in enumerate(self.param_vars.items()):
            tk.Label(grid_f, text=f"{name}:").grid(row=i, column=0, sticky="e", padx=5, pady=4)
            tk.Entry(grid_f, textvariable=var, width=15).grid(row=i, column=1, pady=4)

        self.hyper_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def update_stats(self, *args):
        selected_iids = self.task_tree.selection()
        if not selected_iids: return
        total_count = 0
        for iid in selected_iids:
            actual_idx = int(self.task_tree.item(iid)['values'][0])
            task_path = Path(__file__).parent / self.tasks[actual_idx]['path']
            img_dir = task_path / "images"
            if img_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    total_count += len(list(img_dir.glob(ext)))
                
        ratio = self.train_split_ratio.get()
        train_count = int(total_count * ratio)
        self.stats_lbl.config(text=f"Total: {total_count} | Train: {train_count} | Val: {total_count - train_count}")

    def save_boxes(self):
        if not self.image_list or not self.training_session_path: return
        img_p = self.image_list[self.current_img_idx]
        dw, dh = 1.0 / self.canvas_w, 1.0 / self.canvas_h
        
        labels_dir = self.training_session_path / "labels"
        os.makedirs(labels_dir, exist_ok=True)
        txt_p = labels_dir / f"{img_p.stem}.txt"
        
        with open(txt_p, 'w') as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                cx, cy = ((x1 + x2) / 2.0) * dw, ((y1 + y2) / 2.0) * dh
                nw, nh = abs(x2 - x1) * dw, abs(y2 - y1) * dh
                cx, cy = max(0, min(cx, 1)), max(0, min(cy, 1))
                nw, nh = max(0, min(nw, 1)), max(0, min(nh, 1))
                f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        if self.is_dirty:
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.tasks[idx]['edited'] = now_str
                self.save_json()
                self.task_tree.set(selected[0], column="edited", value=now_str)
            self.is_dirty = False

    def save_json(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.tasks, f, indent=4)

    def update_active_class(self, event):
        self.current_class_id.set(int(self.cls_dropdown.get().split(":")[0]))

    def on_button_press(self, event):
        self.save_snapshot()
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            if abs(event.x - x2) < 10 and abs(event.y - y2) < 10:
                self.selected_box_idx, self.resizing = i, True
                return
        self.start_x, self.start_y = event.x, event.y
        self.drawing_box = self.editor_canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="cyan", width=2)

    def on_move_press(self, event):
        if self.resizing:
            cls, x1, y1, _, _ = self.boxes[self.selected_box_idx]
            self.boxes[self.selected_box_idx] = [cls, x1, y1, event.x, event.y]
            self.redraw()
        elif self.drawing_box:
            self.editor_canvas.coords(self.drawing_box, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        if self.resizing:
            self.resizing = False
            self.save_boxes()
        elif self.drawing_box:
            self.editor_canvas.delete(self.drawing_box)
            self.drawing_box = None

            x1 = max(0, min(self.start_x, self.canvas_w))
            y1 = max(0, min(self.start_y, self.canvas_h))
            x2 = max(0, min(event.x, self.canvas_w))
            y2 = max(0, min(event.y, self.canvas_h))
            
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            if w > 5 and h > 5:
                self.boxes.append([self.current_class_id.get(), x1, y1, x2, y2])
                self.redraw()
                self.save_boxes()
            else:
                self.redraw()

    def on_right_click(self, event):
        self.save_snapshot()
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            if min(x1, x2) < event.x < max(x1, x2) and min(y1, y2) < event.y < max(y1, y2):
                self.boxes.pop(i)
                self.is_dirty = True
                self.redraw()
                self.save_boxes()
                return

    def redraw(self):
        self.editor_canvas.delete("box")
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            color = self.class_colors[int(cls) % len(self.class_colors)]
            self.editor_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
            
            h_size = 6 
            self.editor_canvas.create_rectangle(x2-h_size, y2-h_size, x2+h_size, y2+h_size, 
                                                fill="white", outline="black", tags="box")
            
            cls_name = self.class_names.get(int(cls), f"Class {int(cls)}")
            self.editor_canvas.create_text(x1, y1-10, text=f"{cls_name}", 
                                           fill=color, tags="box", anchor="sw", font=("Arial", 10, "bold"))

    def start_training(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("!", "Select one or more tasks.")
        
        model_pt = self.model_path.get()
        cleaned_params = {}
        for k, v in self.param_vars.items():
            val = v.get().strip()
            if k == "device" and "," in val: cleaned_params[k] = val
            elif val.lower() == "true": cleaned_params[k] = True
            elif val.lower() == "false": cleaned_params[k] = False
            else:
                try:
                    cleaned_params[k] = float(val) if "." in val else int(val)
                except ValueError: cleaned_params[k] = val

        # Provide model directory directly in parameters
        cleaned_params["model_dir"] = self.model_dir.get()

        task_data = []
        for iid in selected_iids:
            actual_idx = int(self.task_tree.item(iid)['values'][0])
            task = self.tasks[actual_idx]
            p = str((Path(__file__).parent / task['path']).absolute())
            w = int(task.get('weight', 1))
            task_data.append((p, w))

        cmd = [
            sys.executable, 
            "train_wrapper.py", 
            model_pt, 
            json.dumps(task_data), 
            json.dumps(cleaned_params), 
            str(self.train_split_ratio.get())
        ]

        self.train_proc = subprocess.Popen(cmd)
        self.log(f"Training started (Standalone Mode). Output target: /model/<date>/run-N. PID: {self.train_proc.pid}")
        self.train_btn.config(state="disabled", text="⌛ Training...")
        threading.Thread(target=self._monitor_training, daemon=True).start()

    def _monitor_training(self):
        while self.train_proc and self.train_proc.poll() is None:
            time.sleep(1)
        self.root.after(0, self._finalize_training_state)

    def _finalize_training_state(self):
        self.train_btn.config(state="normal", text="🔥 START TRAINING")
        self.train_proc = None
        self.log("Training engine finished.")

    def stop_training(self):
        if self.train_proc and self.train_proc.poll() is None:
            if messagebox.askyesno("Confirm", "Stop training?"):
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.train_proc.pid)])
                self.train_proc = None
                self._finalize_training_state()
                self.log("Training terminated.")
  
    def autolabel_current_view(self):
        if not self.image_list: return messagebox.showwarning("!", "No image loaded to label.")
        model_p = self.autolabel_model_path.get()
        if not model_p or not os.path.exists(model_p): return messagebox.showerror("Error", "Select a valid Inference Model (.pt).")

        try:
            self.save_snapshot()
            img_path = self.image_list[self.current_img_idx]
            self.log(f"Running spot-labeling on: {img_path.name}")

            model = YOLO(model_p)
            results = model.predict(source=str(img_path), conf=0.45, save=False)[0]

            self.boxes = []
            updated_classes = False
            
            for box in results.boxes:
                cls = int(box.cls[0])
                if cls not in self.class_names or self.class_names[cls] == "object" or self.class_names[cls].startswith("class_"):
                    self.class_names[cls] = model.names.get(cls, f"class_{cls}")
                    updated_classes = True
                    
                xyxy = box.xyxy[0].tolist() 
                orig_w, orig_h = results.orig_shape 
                scale_x = self.canvas_w / orig_h
                scale_y = self.canvas_h / orig_w

                x1, y1 = xyxy[0] * scale_x, xyxy[1] * scale_y
                x2, y2 = xyxy[2] * scale_x, xyxy[3] * scale_y
                self.boxes.append([cls, x1, y1, x2, y2])

            if updated_classes:
                self.refresh_class_dropdown()
                selected = self.task_tree.selection()
                if selected:
                    idx = int(self.task_tree.item(selected[0])['values'][0])
                    self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                    self.save_json()
                    if self.training_session_path:
                        self.write_data_yaml(self.training_session_path, self.class_names)

            self.redraw()
            self.log("Spot-labeling complete.")
        except Exception as e:
            self.log(f"Spot-labeling Error: {e}")

    def load_previews(self):
        if not self.training_session_path or not self.training_session_path.exists():
            self.log("Error: Task session path not found.")
            return

        images_dir = self.training_session_path / "images"
        self.image_list = []
        if images_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                self.image_list.extend(sorted(list(images_dir.glob(ext))))

        if self.image_list:
            self.current_img_idx = 0
            self.show_image()
            self.log(f"Loaded {len(self.image_list)} images from task.")
        else:
            self.log("No images found in images/ directory.")

    def show_image(self):
        if not self.image_list: return
        img_p = self.image_list[self.current_img_idx]

        with Image.open(img_p) as img_file:
            img = img_file.copy()

        self.canvas_w = 800
        self.canvas_h = int(img.height * (800 / img.width))
        self.tk_img = ImageTk.PhotoImage(img.resize((self.canvas_w, self.canvas_h)))
        self.editor_canvas.config(width=self.canvas_w, height=self.canvas_h)
        self.editor_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.boxes = []
        
        txt_p = (self.training_session_path / "labels" / f"{img_p.stem}.txt")
        updated_classes = False
        if txt_p.exists():
            with open(txt_p, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        c = int(parts[0])
                        if c not in self.class_names:
                            self.class_names[c] = f"class_{c}"
                            updated_classes = True
                    
                        c, cx, cy, nw, nh = map(float, parts)
                        x1, y1 = (cx-nw/2)*self.canvas_w, (cy-nh/2)*self.canvas_h
                        x2, y2 = (cx+nw/2)*self.canvas_w, (cy+nh/2)*self.canvas_h
                        self.boxes.append([c, x1, y1, x2, y2])
                    
        if updated_classes:
            self.refresh_class_dropdown()
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                self.save_json()
                self.write_data_yaml(self.training_session_path, self.class_names)

        self.redraw()
        self.is_dirty = False
        self.idx_lbl.config(text=f"{self.current_img_idx+1}/{len(self.image_list)}")

    def next_img(self): 
        if self.current_img_idx < len(self.image_list)-1: 
            self.save_boxes()
            self.current_img_idx += 1
            self.show_image()
    
    def prev_img(self): 
        if self.current_img_idx > 0: 
            self.save_boxes()
            self.current_img_idx -= 1
            self.show_image()
    
    def browse_model(self):
        init_dir = self.model_dir.get() if Path(self.model_dir.get()).exists() else None
        p = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("YOLO Model", "*.pt")])
        if p:
            self.model_path.set(p)
            self.log(f"Base training model set to: {Path(p).name}")
            m = YOLO(p)
            self.class_names = m.names.copy()
            
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                self.save_json()
                if self.training_session_path:
                    self.write_data_yaml(self.training_session_path, self.class_names)
                
            self.refresh_class_dropdown()

    def browse_autolabel_model(self):
        init_dir = self.model_dir.get() if Path(self.model_dir.get()).exists() else None
        p = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("YOLO Model", "*.pt")])
        if p:
            self.autolabel_model_path.set(p)
            self.log(f"Inference model set to: {Path(p).name}")

    def browse_videos(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if paths:
            for p in paths:
                if p not in self.video_files:
                    self.video_files.append(p)
                    self.v_listbox.insert(tk.END, Path(p).name)
            self.log(f"Added {len(paths)} video(s). Total in queue: {len(self.video_files)}")

    def log(self, msg):
        self.log_area.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_area.see(tk.END)

    def save_snapshot(self):
        self.undo_stack.append([list(box) for box in self.boxes])
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.boxes = self.undo_stack.pop()
            self.redraw()
            self.log("Undo performed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelMakerPro(root)
    root.mainloop()