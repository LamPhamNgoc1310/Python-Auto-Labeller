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
        self.autolabel_model_path = tk.StringVar(value="yolov8s.pt")
        self.export_dir = tk.StringVar(value=str(Path(__file__).parent / "export"))
        self.video_files = []
        
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
        self.extraction_interval = tk.StringVar(value="60")
        self.num_split_parts = tk.StringVar(value="1")

        self.create_widgets()
        self.refresh_task_list()
        
        # Automatic Migration on Startup
        self.migrate_all_legacy_tasks()

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

    # ==========================================
    # YOLO STANDARDIZATION & MIGRATION MODULE
    # ==========================================
    def write_data_yaml(self, task_dir, classes_dict):
        """Generates a standard YOLO data.yaml inside the task root directory."""
        yaml_path = Path(task_dir) / "data.yaml"
        with open(yaml_path, "w") as f:
            f.write("train: images\n")
            f.write("val: images\n\n")
            f.write("names:\n")
            for c_id in sorted([int(k) for k in classes_dict.keys()]):
                f.write(f"  {c_id}: {classes_dict[c_id]}\n")

    def migrate_single_task(self, task_path, task_classes):
        """Deeply searches for any images/labels and forces them into standard YOLO format."""
        task_path = Path(task_path)
        if not task_path.exists():
            return False

        images_dir = task_path / "images"
        labels_dir = task_path / "labels"
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        migrated = False

        # 1. Find all images anywhere inside the task folder (handles deep nesting)
        all_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_images.extend(list(task_path.rglob(ext)))

        for img_p in all_images:
            # Skip if it's already exactly inside the target images/ folder
            if img_p.parent.resolve() == images_dir.resolve():
                continue
            
            # Move Image
            target_img = images_dir / img_p.name
            if not target_img.exists():
                shutil.move(str(img_p), str(target_img))
                migrated = True

            # Move Label (Check right next to the original image path first)
            txt_p = img_p.with_suffix(".txt")
            if txt_p.exists():
                target_txt = labels_dir / txt_p.name
                if not target_txt.exists():
                    shutil.move(str(txt_p), str(target_txt))

        # 2. Aggressive Cleanup: Delete ANY directory that is not 'images' or 'labels'
        for item in task_path.iterdir():
            if item.is_dir() and item.name not in ["images", "labels"]:
                shutil.rmtree(item, ignore_errors=True)
                migrated = True

        # 3. Ensure data.yaml exists
        yaml_path = task_path / "data.yaml"
        if not yaml_path.exists():
            self.write_data_yaml(task_path, task_classes)
            migrated = True

        return migrated

    def migrate_all_legacy_tasks(self):
        """Scans all registered tasks and upgrades legacy directories to standard YOLO layout."""
        count = 0
        for task in self.tasks:
            t_path = Path(__file__).parent / task["path"]
            classes = {int(k): v for k, v in task.get("classes", {0: "object"}).items()}
            if self.migrate_single_task(t_path, classes):
                count += 1
        if count > 0:
            self.log(f"Migrated {count} task(s) to standard YOLO format (images/, labels/, data.yaml).")

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

        # Task Database
        task_frame = tk.LabelFrame(left_panel, text="Task Database", padx=5, pady=5)
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

        # Data Management (Import / Export / Migration)
        data_frame = tk.LabelFrame(left_panel, text="YOLO Import & Export", padx=5, pady=5)
        data_frame.pack(fill="x", pady=5)
        tk.Button(data_frame, text="📦 Import YOLO / CVAT (.zip)", command=self.import_zip, bg="#9b59b6", fg="white").pack(fill="x", pady=2)
        tk.Button(data_frame, text="📤 Export Task as Standard YOLO (.zip)", bg="#2980b9", fg="white", command=self.export_task).pack(fill="x", pady=2)
        tk.Button(data_frame, text="⚡ Run Migration on All Tasks", command=self.manual_migrate, bg="#34495e", fg="white").pack(fill="x", pady=2)

        # Task Actions
        action_frame = tk.LabelFrame(left_panel, text="Task Actions", padx=5, pady=5)
        action_frame.pack(fill="x", pady=5)
        
        btn_row = tk.Frame(action_frame)
        btn_row.pack(fill="x", pady=2)
        tk.Button(btn_row, text="⚖ Weight", bg="#16a085", fg="white", command=self.set_task_weight).pack(side="left", fill="x", expand=True, padx=(0, 2))
        tk.Button(btn_row, text="✏ Rename", command=self.rename_task).pack(side="left", fill="x", expand=True, padx=(2, 0))
        tk.Button(action_frame, text="🗑 Delete Task", bg="#c0392b", fg="white", command=self.delete_task).pack(fill="x", pady=2)

        # Editor Navigation
        nav_frame = tk.LabelFrame(left_panel, text="Editor Navigation", padx=5, pady=5, bg="#34495e", fg="white")
        nav_frame.pack(fill="x", pady=5)
        
        nav_top = tk.Frame(nav_frame, bg="#34495e")
        nav_top.pack(fill="x", pady=2)
        tk.Button(nav_top, text="◀ PREV", command=self.prev_img, width=8).pack(side="left", padx=5)
        self.idx_lbl = tk.Label(nav_top, text="0/0", fg="white", bg="#34495e", font=("Arial", 10, "bold"))
        self.idx_lbl.pack(side="left", expand=True)
        tk.Button(nav_top, text="NEXT ▶", command=self.next_img, width=8).pack(side="right", padx=5)

        tk.Button(nav_frame, text="⚡ SPOT-LABEL CURRENT", bg="#3498db", fg="white", command=self.autolabel_current_view).pack(fill="x", padx=5, pady=5)
        tk.Label(nav_frame, text="Auto-Save Enabled", fg="#2ecc71", bg="#34495e", font=("Arial", 8, "italic")).pack(pady=2)

        # ------------------------------------------
        # MIDDLE PANEL: PIPELINES (AUTO-LABEL & TRAIN)
        # ------------------------------------------
        mid_frame = tk.Frame(paned, padx=10, pady=10)
        paned.add(mid_frame, width=380)

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

    def manual_migrate(self):
        self.migrate_all_legacy_tasks()
        self.refresh_task_list()
        messagebox.showinfo("Migration", "Completed migration scan across all tasks.")

    def build_pipeline_ui(self):
        # Auto-Label Config
        v_lab = tk.LabelFrame(self.pipe_tab, text="Pipeline A: Auto-Labeling", pady=10, padx=10)
        v_lab.pack(fill="x", pady=5)
        
        tk.Label(v_lab, text="Inference Model:").pack(anchor="w")
        m_frame = tk.Frame(v_lab); m_frame.pack(fill="x")
        tk.Entry(m_frame, textvariable=self.autolabel_model_path).pack(side="left", fill="x", expand=True)
        tk.Button(m_frame, text="...", command=self.browse_autolabel_model).pack(side="right")

        tk.Label(v_lab, text="Video Queue:").pack(anchor="w")
        v_list_frame = tk.Frame(v_lab)
        v_list_frame.pack(fill="x", pady=2)

        self.v_listbox = tk.Listbox(v_list_frame, height=5, font=("Arial", 8))
        self.v_listbox.pack(side="left", fill="x", expand=True)
        
        v_scroll = tk.Scrollbar(v_list_frame, orient="vertical", command=self.v_listbox.yview)
        v_scroll.pack(side="right", fill="y")
        self.v_listbox.config(yscrollcommand=v_scroll.set)

        v_btn_f = tk.Frame(v_lab)
        v_btn_f.pack(fill="x", pady=2)
        tk.Button(v_btn_f, text="+ Add Video(s)", command=self.browse_videos).pack(side="left", expand=True, fill="x")
        tk.Button(v_btn_f, text="- Remove Selected", command=self.remove_selected_video).pack(side="left", expand=True, fill="x")
        
        tk.Label(v_lab, text="Extract frames every (sec):").pack(anchor="w")
        tk.Entry(v_lab, textvariable=self.extraction_interval).pack(fill="x", pady=2)

        tk.Label(v_lab, text="Split into N parts:").pack(anchor="w")
        tk.Entry(v_lab, textvariable=self.num_split_parts).pack(fill="x", pady=2)
        self.auto_btn = tk.Button(v_lab, text="⚡ START LABELING", bg="#3498db", fg="white", command=self.start_autolabel)
        self.auto_btn.pack(fill="x", pady=5)

        # Training Config
        d_lab = tk.LabelFrame(self.pipe_tab, text="Pipeline B: Training", pady=10, padx=10)
        d_lab.pack(fill="x", pady=5)
        tk.Label(d_lab, text="BASE MODEL TO TRAIN:").pack(anchor="w")
        m_frame2 = tk.Frame(d_lab)
        m_frame2.pack(fill="x", pady=2)
        tk.Entry(m_frame2, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        tk.Button(m_frame2, text="...", command=self.browse_model).pack(side="right", padx=2)

        self.train_btn = tk.Button(d_lab, text="🔥 START TRAINING", bg="#e67e22", fg="white", height=2, command=self.start_training)
        self.train_btn.pack(fill="x", pady=10)
        
        tk.Button(d_lab, text="🛑 CANCEL TRAINING", bg="#c0392b", fg="white", command=self.stop_training).pack(fill="x", pady=2)

        tk.Label(self.pipe_tab, text="Activity Log:").pack(anchor="w", pady=(10, 0))
        self.log_area = scrolledtext.ScrolledText(self.pipe_tab, height=10, font=("Consolas", 8))
        self.log_area.pack(fill="both", expand=True)

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
        new_name = simpledialog.askstring("Add Class", "Enter the name of the new class:")
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
        
        warning_msg = f"Delete class '{curr_id}: {curr_name}'?\n\nNOTE: Any boxes in the CURRENT image using this class will be removed."
        if messagebox.askyesno("Delete Class", warning_msg):
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
            self.log(f"Set weight to {new_w}x for {len(selected)} tasks.")

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

    # ==========================================
    # STANDARDIZED YOLO IMPORT & EXPORT
    # ==========================================
    def export_task(self):
        """Directly packages the standard images/, labels/, and data.yaml into a YOLO zip."""
        if not self.training_session_path:
            return messagebox.showwarning("!", "Select a task to export first.")
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".zip", 
            initialfile=f"{self.training_session_path.name}_yolo.zip",
            filetypes=[("Zip files", "*.zip")]
        )
        if not file_path:
            return

        try:
            self.log("Exporting task in standard YOLO format...")
            
            # Ensure data.yaml reflects latest class configuration
            self.write_data_yaml(self.training_session_path, self.class_names)
            
            # Create zip package
            base_name = str(file_path).replace('.zip', '')
            shutil.make_archive(base_name, 'zip', self.training_session_path)
            
            self.log(f"Exported YOLO archive to: {file_path}")
            messagebox.showinfo("Export Complete", "Standard YOLO dataset package exported successfully!")
        except Exception as e:
            self.log(f"YOLO Export Failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def import_zip(self):
        """Extracts any dataset archive and standardizes it into images/, labels/, and data.yaml."""
        paths = filedialog.askopenfilenames(filetypes=[("Zip files", "*.zip")])
        if not paths: return

        for p in paths:
            p = Path(p)
            ts = datetime.now().strftime("%H%M%S_%f")[:10]
            extract_to = Path(self.export_dir.get()) / f"yolo_{ts}"
            os.makedirs(extract_to, exist_ok=True)
            
            self.log(f"Importing and standardizing YOLO package: {p.name}...")
            try:
                with zipfile.ZipFile(p, 'r') as z:
                    z.extractall(extract_to)
                
                images_dir = extract_to / "images"
                labels_dir = extract_to / "labels"
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                # Locate all image files anywhere in the package
                all_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    all_images.extend(list(extract_to.rglob(ext)))
                
                # Exclude images that are already inside our target images_dir
                all_images = [img for img in all_images if img.parent != images_dir]

                if not all_images and not list(images_dir.glob("*.jpg")):
                    self.log(f"No images found in {p.name}")
                    shutil.rmtree(extract_to, ignore_errors=True)
                    continue

                for img_path in all_images:
                    dest_img = images_dir / img_path.name
                    if not dest_img.exists():
                        shutil.move(str(img_path), str(dest_img))
                    
                    # Look for matching .txt label
                    txt_name = img_path.stem + ".txt"
                    found_txts = [t for t in extract_to.rglob(txt_name) if t.parent != labels_dir]
                    if found_txts:
                        dest_txt = labels_dir / found_txts[0].name
                        if not dest_txt.exists():
                            shutil.move(str(found_txts[0]), str(dest_txt))

                # Parse classes from data.yaml / dataset.yaml / classes.txt
                task_classes = {}
                data_yamls = list(extract_to.rglob("data.yaml")) + list(extract_to.rglob("dataset.yaml"))
                if data_yamls:
                    with open(data_yamls[0], 'r') as f:
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
                                    c_name = stripped.replace("-", "", 1).strip().strip("\"'")
                                    task_classes[len(task_classes)] = c_name
                                elif ":" in stripped:
                                    parts = stripped.split(":", 1)
                                    if parts[0].strip().isdigit():
                                        c_id = int(parts[0].strip())
                                        c_name = parts[1].strip().strip("\"'")
                                        task_classes[c_id] = c_name

                if not task_classes:
                    classes_txts = list(extract_to.rglob("classes.txt"))
                    if classes_txts:
                        with open(classes_txts[0], 'r') as f:
                            task_classes = {i: line.strip() for i, line in enumerate(f) if line.strip()}
                
                if not task_classes:
                    task_classes = {0: "object"}
                
                # Cleanup stray leftover directories
                for item in extract_to.iterdir():
                    if item.is_dir() and item.name not in ["images", "labels"]:
                        shutil.rmtree(item, ignore_errors=True)
                    elif item.is_file() and item.name not in ["data.yaml"]:
                        os.remove(item)

                # Write canonical root data.yaml
                self.write_data_yaml(extract_to, task_classes)

                task_name = p.stem 
                try: rel_path = extract_to.relative_to(Path(__file__).parent)
                except ValueError: rel_path = extract_to

                self.save_task_to_db(task_name, rel_path, "import", classes=task_classes)
                self.log(f"Imported standardized YOLO Task: {task_name} ({len(task_classes)} classes)")

            except Exception as e:
                self.log(f"YOLO Import Failed: {e}")
        
        self.refresh_task_list()

    def delete_task(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("!", "Please select a task to delete.")

        warning_msg = f"PERMANENTLY remove {len(selected_iids)} task(s)?\n\nWARNING: This will delete folders and images from disk!"
        if messagebox.askyesno("Confirm Delete", warning_msg):
            indices_to_remove = []
            for iid in selected_iids:
                actual_idx = int(self.task_tree.item(iid)['values'][0])
                indices_to_remove.append(actual_idx)
            
            indices_to_remove.sort(reverse=True)
            for idx in indices_to_remove:
                if 0 <= idx < len(self.tasks):
                    try:
                        task_path = Path(__file__).parent / self.tasks[idx]['path']
                        if task_path.exists() and task_path.is_dir():
                            shutil.rmtree(task_path, ignore_errors=True)
                            self.log(f"Deleted from disk: {task_path.name}")
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
                self.log("Active task was deleted. Editor cleared.")

    def remove_selected_video(self):
        selection = self.v_listbox.curselection()
        if not selection: return
        for index in reversed(selection):
            self.video_files.pop(index)
            self.v_listbox.delete(index)
        self.log(f"Removed selection. Remaining: {len(self.video_files)}")

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
                                orient="horizontal", variable=self.train_split_ratio,
                                command=self.update_stats)
        split_slider.pack(side="left", fill="x", expand=True, padx=5)

        self.stats_lbl = tk.Label(stats_frame, text="Total: 0 | Train: 0 | Val: 0", 
                                  fg="#3498db", font=("Arial", 9, "bold"))
        self.stats_lbl.pack(fill="x", pady=5)

        tk.Label(self.scrollable_frame, text="YOLO Parameters", font=("Arial", 10, "bold")).pack(pady=10)
        grid_f = tk.Frame(self.scrollable_frame); grid_f.pack(padx=20)
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
            item_data = self.task_tree.item(iid)
            actual_idx = int(item_data['values'][0])
            task_path = Path(__file__).parent / self.tasks[actual_idx]['path']
            
            # Deep search matching migration logic
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                total_count += len(list(task_path.rglob(ext)))
                
        ratio = self.train_split_ratio.get()
        train_count = int(total_count * ratio)
        self.stats_lbl.config(text=f"Total: {total_count} | Train: {train_count} | Val: {total_count - train_count}")

    # ==========================================
    # EDITOR STORAGE (YOLO FORMAT)
    # ==========================================
    def save_boxes(self):
        """Saves bounding boxes to labels/<image_name>.txt in standard YOLO normalized format."""
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
                self.log("Discarded tiny box.")

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

    def start_autolabel(self):
        if not self.autolabel_model_path.get(): return messagebox.showwarning("!", "Select an Inference Model!")
        threading.Thread(target=self._exec_labeling, daemon=True).start()

    def _exec_labeling(self):
        self.auto_btn.config(state="disabled")
        try:
            interval = float(self.extraction_interval.get())
            parts_count = int(self.num_split_parts.get())
            
            session = AutoLabelEngine.run_auto_label_process(
                self.autolabel_model_path.get(), 
                self.video_files, 
                self.export_dir.get(), 
                frame_interval=interval, 
                log_callback=self.log
            )
            
            if parts_count > 1:
                created_parts = AutoLabelEngine.split_session_into_parts(session, parts_count, log_callback=self.log)
                for p_name, p_path in created_parts:
                    self.migrate_single_task(p_path, self.class_names)
                    self.save_task_to_db(p_name, p_path, "auto_part", classes=self.class_names)
                self.log(f"Session split into {len(created_parts)} standardized YOLO tasks.")
            else:
                self.migrate_single_task(session, self.class_names)
                self.save_task_to_db(session.name, session, "auto", classes=self.class_names)
                self.training_session_path = session
                self.load_previews()
                
        except Exception as e:
            self.log(f"Auto-Label Task Failed: {e}")
        finally:
            self.auto_btn.config(state="normal")
            self.refresh_task_list()

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
        self.log(f"Training started (Standalone Mode). PID: {self.train_proc.pid}")
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
        if not model_p or not os.path.exists(model_p): return messagebox.showerror("Error", "Select an Inference Model.")

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
        """Loads images smoothly matching the exact paths from migration."""
        if not self.training_session_path or not self.training_session_path.exists():
            self.log("Error: Task session path not found.")
            return

        self.image_list = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            self.image_list.extend(sorted(list(self.training_session_path.rglob(ext))))

        if self.image_list:
            self.current_img_idx = 0
            self.show_image()
            self.log(f"Loaded {len(self.image_list)} images from YOLO workspace.")
        else:
            self.log("No images found in images/ directory.")

    def show_image(self):
        if not self.image_list: return
        img_p = self.image_list[self.current_img_idx]
        img = Image.open(img_p)
        self.canvas_w = 800
        self.canvas_h = int(img.height * (800 / img.width))
        self.tk_img = ImageTk.PhotoImage(img.resize((self.canvas_w, self.canvas_h)))
        self.editor_canvas.config(width=self.canvas_w, height=self.canvas_h)
        self.editor_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.boxes = []
        
        # Load from labels/<stem>.txt
        txt_p = (self.training_session_path / "labels" / f"{img_p.stem}.txt")
        if not txt_p.exists():
            # Fallback to local sibling label
            txt_p = img_p.with_suffix('.txt')

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
        p = filedialog.askopenfilename()
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
            self.log("Classes updated from model.")

    def browse_autolabel_model(self):
        p = filedialog.askopenfilename()
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
            self.log(f"Added {len(paths)} video(s). Total: {len(self.video_files)}")

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