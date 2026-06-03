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
        self.model_path = tk.StringVar()
        self.autolabel_model_path = tk.StringVar()
        self.export_dir = tk.StringVar(value=str(Path(__file__).parent / "export"))
        self.video_files = []
        
        # just added
        self.db_path = Path(__file__).parent / "tasks_db.json"
        self.tasks = self.load_tasks_db()
        # Add to your __init__ variables
        self.oversample_factor = tk.StringVar(value="1")

        self.training_session_path = None
        self.image_list = []
        self.current_img_idx = 0

        self.class_names = {0: "object"} 
        self.current_class_id = tk.IntVar(value=0)
        # Add a diverse color palette for dynamically created classes
        self.class_colors = [
            "#2ecc71", "#e74c3c", "#3498db", "#f1c40f", 
            "#9b59b6", "#e67e22", "#1abc9c", "#e84393", "#34495e"
        ]

        self.undo_stack = [] # Stores snapshots of self.boxes
        # Bind the Ctrl+Z key to the root window
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.is_dirty = False # NEW: Tracks if the current image has been edited
        self.model_path = tk.StringVar(value="yolov8s.pt") # Better default
        self.autolabel_model_path = tk.StringVar(value="yolov8s.pt")
        
        # --- Editor State ---
        self.boxes = [] # [cls, x1, y1, x2, y2]
        self.selected_box_idx = -1
        self.drawing_box = None
        self.resizing = False
        self.train_proc = None

        # --- Hyperparameters ---
        self.train_params = {
            # "epochs": 500,               # Increased to give the model more time to converge
            # "batch": 16,
            # "imgsz": 640,
            # "patience": 100,             # Increased so it doesn't quit too early during fine-tuning
            # "optimizer": "SGD",          # Switching to SGD often yields better final mAP than 'auto' (AdamW)
            # "lr0": 0.01,
            # "lrf": 0.001,                # Lower final LR (0.001 vs 0.01) helps the model "settle" into the global minimum
            # "momentum": 0.937,
            # "weight_decay": 0.0005,
            # "warmup_epochs": 3.0,        # Reduced warmup; 30 is very high for 300-500 epochs
            # "cos_lr": "True",

            # # Loss Gains - Focused on Box Precision
            # "box": 10.0,                 # Slightly increased to prioritize IoU accuracy
            # "cls": 0.5,
            # "dfl": 2.0,                  # Increased DFL helps the model refine box boundaries (critical for mAP95)

            # # Augmentation - Slightly dialed back to favor precision in later epochs
            # "degrees": 0.0,              # Rotation can sometimes make tight box fitting harder
            # "translate": 0.1,            # Reduced from 0.3 to keep objects more centered/stable
            # "scale": 0.5,
            # "shear": 0.0,
            # "perspective": 0.0,
            # "flipud": 0.0,
            # "fliplr": 0.5,
            # "mosaic": 1.0,               # Keep mosaic on
            # "mixup": 0.0,                # Set to 0.0 unless your dataset is massive; can introduce noise
            # "copy_paste": 0.0,
            
            # "hsv_h": 0.015,
            # "hsv_s": 0.7,
            # "hsv_v": 0.4,
            # "close_mosaic": 10           # NEW: Disables mosaic for the last 10 epochs to refine coordinates
            "epochs": 100,
            "batch": 16,
            "patience": 30,      # Don't stop early; give mAP95 time to climb
            "workers": 2,
            "cos_lr": "true",
            "lr0": 0.01,         # Lower for fine-tuning 100 images
            "lrf": 0.01,
            "imgsz": 640,
            "device": "-1",
            
            # Precision Boosters
            "box": 10.0,          # High box priority
            "dfl": 2.5,           # High edge precision
            "close_mosaic": 20,   # Critical: disables heavy augmentation at the end
            "freeze": 0,

            # "Slight" Augmentation (Best for fine-tuning)
            "mosaic": 1.0,        # Keep it on for the first 285 epochs
            "mixup": 0.8,
            "fliplr": 0.5,        # Standard
            "scale": 0.5,         # Slightly reduced from 0.5 to keep objects stable
            "translate": 0.1,     # Reduced to prevent objects from clipping out
            "degrees": 12,
            "perspective": 0.001,
            "shear" : 2.0,

            "hsv_h": 0.015,
            "hsv_s": 0.25,
            "hsv_v": 0.3,
        }
        self.param_vars = {k: tk.StringVar(value=v) for k, v in self.train_params.items()}
        self.train_split_ratio = tk.DoubleVar(value=0.85)
        self.extraction_interval = tk.StringVar(value="60") # Default to 5 seconds
        self.num_split_parts = tk.StringVar(value="1") # 1 means no split

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

    def create_widgets(self):
        # Global Header - Simplified
        header = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header.pack(fill="x")
        tk.Label(header, text="YOLO ACTIVE LEARNING SUITE", fg="white", 
                 bg="#2c3e50", font=("Arial", 12, "bold")).pack()

        # BIGGER ORGANISER BORDERS (sashwidth=12)
        paned = tk.PanedWindow(self.root, orient="horizontal", 
                               sashwidth=12, sashpad=2, 
                               bg="#7f8c8d", showhandle=True)
        paned.pack(fill="both", expand=True)

        # --- TASK MANAGER PANEL (FAR LEFT) ---
        task_frame = tk.LabelFrame(paned, text="Task Manager", padx=5, pady=5)
        paned.add(task_frame, width=450)

        cols = ("idx", "name", "weight", "created", "edited")
        self.task_tree = ttk.Treeview(task_frame, columns=cols, show="headings", selectmode="extended")
        
        self.task_tree.heading("idx", text="#")
        self.task_tree.heading("name", text="Name")
        self.task_tree.heading("weight", text="Weight (x)")
        self.task_tree.heading("created", text="Time Added")
        self.task_tree.heading("edited", text="Last Edit")
        
        self.task_tree.column("idx", width=30, anchor="center")
        self.task_tree.column("name", width=140)
        self.task_tree.column("weight", width=70, anchor="center")
        self.task_tree.column("created", width=110)
        self.task_tree.column("edited", width=110)
        
        self.task_tree.pack(fill="both", expand=True)
        self.task_tree.bind("<<TreeviewSelect>>", self.on_task_select)

        # Task Controls
        t_btn_f = tk.Frame(task_frame)
        t_btn_f.pack(fill="x")
        tk.Button(t_btn_f, text="⚖ SET WEIGHT (Multiplier)", bg="#16a085", fg="white", command=self.set_task_weight).pack(fill="x", pady=2)
        tk.Button(t_btn_f, text="Rename Task", command=self.rename_task).pack(fill="x", pady=2)
        tk.Button(t_btn_f, text="Export (.zip)", bg="#2980b9", fg="white", command=self.export_task).pack(fill="x", pady=2)
        tk.Button(t_btn_f, text="Delete Task", bg="#c0392b", fg="white", command=self.delete_task).pack(fill="x", pady=2)

        # --- MIDDLE PANEL: PIPELINES ---
        left_frame = tk.Frame(paned, padx=10, pady=10)
        paned.add(left_frame, width=420)

        self.ctrl_tabs = ttk.Notebook(left_frame)
        self.ctrl_tabs.pack(fill="both", expand=True)

        self.pipe_tab = tk.Frame(self.ctrl_tabs, padx=5, pady=5)
        self.hyper_tab = tk.Frame(self.ctrl_tabs, padx=5, pady=5)
        self.ctrl_tabs.add(self.pipe_tab, text="Pipelines")
        self.ctrl_tabs.add(self.hyper_tab, text="Hyperparams")

        self.build_pipeline_ui()
        self.build_scrollable_hyper_tab()

        # --- RIGHT PANEL: EDITOR ---
        right_frame = tk.Frame(paned, bg="#34495e")
        paned.add(right_frame)

        cls_f = tk.Frame(right_frame, bg="#34495e", pady=5)
        cls_f.pack(fill="x")
        tk.Label(cls_f, text="Active Class:", fg="white", bg="#34495e").pack(side="left", padx=10)
        
        # Combobox
        self.cls_dropdown = ttk.Combobox(cls_f, state="readonly", width=15)
        self.cls_dropdown.pack(side="left", padx=5)
        self.cls_dropdown.bind("<<ComboboxSelected>>", self.update_active_class)
        
        # New Add/Delete Buttons
        tk.Button(cls_f, text="+ Add", bg="#2ecc71", fg="white", font=("Arial", 8, "bold"), 
                  command=self.add_class).pack(side="left", padx=5)
        tk.Button(cls_f, text="- Del", bg="#e74c3c", fg="white", font=("Arial", 8, "bold"), 
                  command=self.delete_class).pack(side="left", padx=5)
        
        # Initialize the dropdown
        self.refresh_class_dropdown()

        self.editor_canvas = tk.Canvas(right_frame, bg="#1e1e1e", cursor="cross")
        self.editor_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.editor_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.editor_canvas.bind("<B1-Motion>", self.on_move_press)
        self.editor_canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.editor_canvas.bind("<Button-3>", self.on_right_click)

        nav = tk.Frame(right_frame, bg="#34495e", pady=10)
        nav.pack(fill="x")
        tk.Button(nav, text="◀ PREV", command=self.prev_img).pack(side="left", padx=20)
        self.idx_lbl = tk.Label(nav, text="0/0", fg="white", bg="#34495e"); self.idx_lbl.pack(side="left", expand=True)
        tk.Label(nav, text="Auto-Save Enabled", fg="#2ecc71", bg="#34495e", font=("Arial", 8, "italic")).pack(side="left", padx=10)
        tk.Button(nav, text="⚡ SPOT-LABEL", bg="#3498db", fg="white", 
          command=self.autolabel_current_view).pack(side="left", padx=5)
        tk.Button(nav, text="NEXT ▶", command=self.next_img).pack(side="right", padx=20)
    
    
    def refresh_class_dropdown(self):
        """Rebuilds the dropdown list based on self.class_names"""
        values = [f"{k}: {v}" for k, v in self.class_names.items()]
        self.cls_dropdown['values'] = values
        
        if values:
            if self.current_class_id.get() in self.class_names:
                # If current class still exists, keep it selected
                idx = list(self.class_names.keys()).index(self.current_class_id.get())
                self.cls_dropdown.current(idx)
            else:
                # Default to the first item
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
            
            original_len = len(self.boxes)
            self.boxes = [b for b in self.boxes if int(b[0]) != curr_id]
            if len(self.boxes) < original_len:
                self.is_dirty = True
                self.save_boxes()
                self.redraw()
            
            self.refresh_class_dropdown()
            self.log(f"Class deleted: {curr_name}")

    def refresh_task_list(self):
        """Clears and repopulates the Task Manager including Weight."""
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
            
        for i, task in enumerate(self.tasks):
            # Fallback to weight 1 if not exists
            t_weight = task.get('weight', 1) 
            self.task_tree.insert("", "end", values=(
                i, 
                task['name'], 
                f"{t_weight}x", 
                task.get('date', "N/A"), 
                task.get('edited', "N/A")
            ))

    def set_task_weight(self):
        """Allows user to change the oversampling weight of selected tasks."""
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
        
        # Pull saved indices back into integers
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
        
        # FIX: Get index from values[0] instead of int(iid)
        idx = int(self.task_tree.item(selected_iids[0])['values'][0])
        old_name = self.tasks[idx]['name']
        new_name = tk.simpledialog.askstring("Rename", "New task name:", initialvalue=old_name)
        
        if new_name:
            self.tasks[idx]['name'] = new_name
            self.tasks[idx]['edited'] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.save_json() 
            self.refresh_task_list()

    def export_task(self):
        if not self.training_session_path:
            return messagebox.showwarning("!", "Select a task to export first.")
        
        # Ask where to save the CVAT formatted ZIP
        file_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile=f"{self.training_session_path.name}_cvat1_1.zip")
        if not file_path:
            return

        try:
            self.log("Packaging task to CVAT YOLO 1.1 Standard...")
            
            # 1. Create a temporary staging directory
            temp_cvat_dir = Path(self.export_dir.get()) / f"temp_cvat_{datetime.now().strftime('%H%M%S')}"
            obj_train_data_dir = temp_cvat_dir / "obj_train_data"
            os.makedirs(obj_train_data_dir, exist_ok=True)
            
            # Gather all images/labels from the current task's obj_train_data* folders
            data_folders = [d for d in self.training_session_path.glob("obj_train_data*") if d.is_dir()]
            train_txt_path = temp_cvat_dir / "train.txt"
            
            max_class_id = 0 # Tracks highest class ID used
            
            # 2. Build obj_train_data and train.txt
            with open(train_txt_path, "w") as train_f:
                for folder in data_folders:
                    for img_path in folder.glob("*.jpg"):
                        # Copy image
                        shutil.copy(img_path, obj_train_data_dir / img_path.name)
                        
                        txt_path = img_path.with_suffix('.txt')
                        if txt_path.exists():
                            shutil.copy(txt_path, obj_train_data_dir / txt_path.name)
                            # Parse labels to find the highest class ID for obj.names
                            with open(txt_path, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        c_id = int(parts[0])
                                        if c_id > max_class_id:
                                            max_class_id = c_id
                                            
                        # Write path string required by CVAT 1.1 format
                        train_f.write(f"data/obj_train_data/{img_path.name}\n")
            
            # 3. Build obj.names
            with open(temp_cvat_dir / "obj.names", "w") as f:
                for i in range(max_class_id + 1):
                    # Use class names from loaded model if present, otherwise generic
                    name = self.class_names.get(i, f"class_{i}")
                    f.write(f"{name}\n")
                    
            # 4. Build obj.data
            with open(temp_cvat_dir / "obj.data", "w") as f:
                f.write(f"classes = {max_class_id + 1}\n")
                f.write("train = data/train.txt\n")
                f.write("names = data/obj.names\n")
                f.write("backup = backup/\n")
                
            # 5. Zip and cleanup
            base_name = str(file_path).replace('.zip', '')
            shutil.make_archive(base_name, 'zip', temp_cvat_dir)
            shutil.rmtree(temp_cvat_dir, ignore_errors=True)
            
            self.log(f"Exported CVAT 1.1 zip to: {file_path}")
            messagebox.showinfo("Export Complete", "CVAT YOLO 1.1 package generated successfully!")
            
        except Exception as e:
            self.log(f"CVAT Export Failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def delete_task(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("!", "Please select a task to delete.")

        # Updated Warning Message
        warning_msg = f"PERMANENTLY remove {len(selected_iids)} task(s)?\n\nWARNING: This will delete the folders and images from your hard drive!"
        if messagebox.askyesno("Confirm Delete", warning_msg):
            
            indices_to_remove = []
            for iid in selected_iids:
                actual_idx = int(self.task_tree.item(iid)['values'][0])
                indices_to_remove.append(actual_idx)
            
            # Sort reverse to avoid list-shifting issues during pop()
            indices_to_remove.sort(reverse=True)
            
            for idx in indices_to_remove:
                if 0 <= idx < len(self.tasks):
                    # --- NEW: Delete from physical disk ---
                    try:
                        task_path = Path(__file__).parent / self.tasks[idx]['path']
                        if task_path.exists() and task_path.is_dir():
                            # forcefully remove the directory and all contents
                            shutil.rmtree(task_path, ignore_errors=True)
                            self.log(f"Deleted from disk: {task_path.name}")
                    except Exception as e:
                        self.log(f"Disk Delete Error: {e}")
                    # --------------------------------------
                    
                    # Remove from Database
                    self.tasks.pop(idx)
            
            self.save_json()
            self.refresh_task_list()
            
            # --- NEW: Clear Editor if active task was deleted ---
            if self.training_session_path and not self.training_session_path.exists():
                self.training_session_path = None
                self.image_list = []
                self.boxes = []
                self.editor_canvas.delete("all")
                self.idx_lbl.config(text="0/0")
                self.log("Active task was deleted. Editor cleared.")
            # ----------------------------------------------------
            
            self.log(f"Successfully deleted {len(indices_to_remove)} task(s) from history and disk.")
    
    
    def build_pipeline_ui(self):
        # Auto-Label Config
        v_lab = tk.LabelFrame(self.pipe_tab, text="Pipeline A: Auto-Labeling", pady=10, padx=10)
        v_lab.pack(fill="x", pady=5)
        
        tk.Label(v_lab, text="Inference Model:").pack(anchor="w")
        m_frame = tk.Frame(v_lab); m_frame.pack(fill="x")
        tk.Entry(m_frame, textvariable=self.autolabel_model_path).pack(side="left", fill="x", expand=True)
        tk.Button(m_frame, text="...", command=self.browse_autolabel_model).pack(side="right")

        # --- Video Selection List ---
        tk.Label(v_lab, text="Video Queue:").pack(anchor="w")
        v_list_frame = tk.Frame(v_lab)
        v_list_frame.pack(fill="x", pady=2)

        self.v_listbox = tk.Listbox(v_list_frame, height=5, font=("Arial", 8))
        self.v_listbox.pack(side="left", fill="x", expand=True)
        
        v_scroll = tk.Scrollbar(v_list_frame, orient="vertical", command=self.v_listbox.yview)
        v_scroll.pack(side="right", fill="y")
        self.v_listbox.config(yscrollcommand=v_scroll.set)

        # Video Action Buttons
        v_btn_f = tk.Frame(v_lab)
        v_btn_f.pack(fill="x", pady=2)
        tk.Button(v_btn_f, text="+ Add Video(s)", command=self.browse_videos).pack(side="left", expand=True, fill="x")
        tk.Button(v_btn_f, text="- Remove Selected", command=self.remove_selected_video).pack(side="left", expand=True, fill="x")
        
        
        # Inside build_pipeline_ui, under the "Select Videos" button
        tk.Label(v_lab, text="Extract frames every (sec):").pack(anchor="w")
        tk.Entry(v_lab, textvariable=self.extraction_interval).pack(fill="x", pady=2)

        # Under the extraction interval field
        tk.Label(v_lab, text="Split into N parts:").pack(anchor="w")
        tk.Entry(v_lab, textvariable=self.num_split_parts).pack(fill="x", pady=2)
        self.auto_btn = tk.Button(v_lab, text="⚡ START LABELING", bg="#3498db", fg="white", command=self.start_autolabel)
        self.auto_btn.pack(fill="x", pady=5)

        # Dataset Config
        d_lab = tk.LabelFrame(self.pipe_tab, text="Pipeline B: Training", pady=10, padx=10)
        d_lab.pack(fill="x", pady=5)
        tk.Label(d_lab, text="BASE MODEL TO TRAIN:").pack(anchor="w")
        m_frame = tk.Frame(d_lab)
        m_frame.pack(fill="x", pady=2)
        tk.Entry(m_frame, textvariable=self.model_path).pack(side="left", fill="x", expand=True)
        tk.Button(m_frame, text="...", command=self.browse_model).pack(side="right", padx=2)

        tk.Button(d_lab, text="📦 Import CVAT Zip", command=self.import_zip, bg="#9b59b6", fg="white").pack(fill="x", pady=2)
        tk.Button(d_lab, text="🔄 Load Recent Session", command=self.load_recent_session).pack(fill="x", pady=2)
        

        
        self.train_btn = tk.Button(self.pipe_tab, text="🔥 START TRAINING", bg="#e67e22", fg="white", height=2, command=self.start_training)
        self.train_btn.pack(fill="x", pady=10)
        
        tk.Button(self.pipe_tab, text="🛑 CANCEL TRAINING", bg="#c0392b", fg="white", command=self.stop_training).pack(fill="x", pady=2)

        self.log_area = scrolledtext.ScrolledText(self.pipe_tab, height=15, font=("Consolas", 8))
        self.log_area.pack(fill="both", expand=True)

    def remove_selected_video(self):
        """Removes highlighted videos from both the UI and the internal path list."""
        selection = self.v_listbox.curselection()
        if not selection:
            return
        
        # We delete in reverse order so the indices don't shift while we are working
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
        
        # Mousewheel
        self.hyper_tab.bind_all("<MouseWheel>", lambda e: self.hyper_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # --- DATA SPLIT SECTION (ADD THIS) ---
        stats_frame = tk.LabelFrame(self.scrollable_frame, text="Dataset Distribution", padx=10, pady=10)
        stats_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(stats_frame, text="Train/Val Split Ratio:").pack(side="left")
        
        # Slider updates the stats live
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
        """Calculates Train/Val counts from all matching data folders."""
        selected_iids = self.task_tree.selection()
        if not selected_iids: return
        
        total_count = 0
        for iid in selected_iids:
            # FIX: Get the actual numeric index from the Treeview column 'idx'
            item_data = self.task_tree.item(iid)
            actual_idx = int(item_data['values'][0]) # This is your 0, 1, 2...
            
            task_path = Path(__file__).parent / self.tasks[actual_idx]['path']
            
            # Use flexible search for folders like obj_train_data_part1, etc.
            data_folders = [d for d in task_path.glob("obj_train_data*") if d.is_dir()]
            for folder in data_folders:
                total_count += len(list(folder.glob("*.jpg")))
                
        ratio = self.train_split_ratio.get()
        train_count = int(total_count * ratio)
        self.stats_lbl.config(text=f"Total: {total_count} | Train: {train_count} | Val: {total_count - train_count}")

    # --- EDITOR ACTIONS ---
    def save_boxes(self):
        """Saves boxes but only updates timestamp if is_dirty is True."""
        if not self.image_list: return
        img_p = self.image_list[self.current_img_idx]
        dw, dh = 1.0/self.canvas_w, 1.0/self.canvas_h
        
        # Always write the text file (Auto-save)
        with open(img_p.with_suffix('.txt'), 'w') as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                cx, cy = ((x1 + x2) / 2.0) * dw, ((y1 + y2) / 2.0) * dh
                nw, nh = abs(x2 - x1) * dw, abs(y2 - y1) * dh
                cx, cy = max(0, min(cx, 1)), max(0, min(cy, 1))
                nw, nh = max(0, min(nw, 1)), max(0, min(nh, 1))
                f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        # NEW: Only update DB timestamp if an actual edit occurred
        if self.is_dirty:
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.tasks[idx]['edited'] = now_str
                self.save_json()
                self.task_tree.set(selected[0], column="edited", value=now_str)
            self.is_dirty = False # Reset flag after saving

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
            self.save_boxes() # Auto-save after a resize
        elif self.drawing_box:
            # 1. ALWAYS delete the cyan ghost box from the canvas immediately
            self.editor_canvas.delete(self.drawing_box)
            self.drawing_box = None

            # 2. CLAMP: Snaps coordinates to the image boundaries
            # This prevents "out of bounds" errors in YOLO
            x1 = max(0, min(self.start_x, self.canvas_w))
            y1 = max(0, min(self.start_y, self.canvas_h))
            x2 = max(0, min(event.x, self.canvas_w))
            y2 = max(0, min(event.y, self.canvas_h))
            
            # 3. SIZE FILTER: Calculate width and height
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            # Only save if it's a real box (at least 5x5 pixels)
            if w > 5 and h > 5:
                # Store the box
                self.boxes.append([self.current_class_id.get(), x1, y1, x2, y2])
                self.redraw()
                self.save_boxes() # Auto-save the new box
            else:
                # If it's a "dot", we just redraw to keep things clean
                self.redraw()
                self.log("Discarded accidental dot/tiny box.")

    def on_right_click(self, event):
        self.save_snapshot()
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            if min(x1, x2) < event.x < max(x1, x2) and min(y1, y2) < event.y < max(y1, y2):
                self.boxes.pop(i)
                self.is_dirty = True # NEW: Mark as edited
                self.redraw()
                self.save_boxes()
                return

    
    def redraw(self):
        self.editor_canvas.delete("box")
        for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
            # Pick a color dynamically from the palette based on the class ID
            color = self.class_colors[int(cls) % len(self.class_colors)]
            
            self.editor_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
            
            h_size = 6 
            self.editor_canvas.create_rectangle(x2-h_size, y2-h_size, x2+h_size, y2+h_size, 
                                                fill="white", outline="black", tags="box")
            
            # Show the actual class Name instead of just the number
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
            
            # 1. Run standard auto-label
            session = AutoLabelEngine.run_auto_label_process(
                self.autolabel_model_path.get(), 
                self.video_files, 
                self.export_dir.get(), 
                frame_interval=interval, 
                log_callback=self.log
            )
            
            # 2. Handle Splitting
            if parts_count > 1:
                # This now returns a list of (name, path) for each part
                created_parts = AutoLabelEngine.split_session_into_parts(session, parts_count, log_callback=self.log)
                
                for p_name, p_path in created_parts:
                    self.save_task_to_db(p_name, p_path, "auto_part")
                
                self.log(f"Session split into {len(created_parts)} tasks.")
            else:
                # Standard single session save
                self.save_task_to_db(session.name, session, "auto")
                self.training_session_path = session
                self.load_previews()
                
        except Exception as e:
            self.log(f"Auto-Label Task Failed: {e}")
        finally:
            self.auto_btn.config(state="normal")

    def start_training(self):
        selected_iids = self.task_tree.selection()
        if not selected_iids:
            return messagebox.showwarning("!", "Select one or more tasks.")
        
        model_pt = self.model_path.get()
        
        # 1. Clean parameters (Ensure they are JSON serializable)
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

        # 2. Gather task data (Path and Weight)
        task_data = []
        for iid in selected_iids:
            actual_idx = int(self.task_tree.item(iid)['values'][0])
            task = self.tasks[actual_idx]
            p = str((Path(__file__).parent / task['path']).absolute())
            w = int(task.get('weight', 1))
            task_data.append((p, w))

        # 3. Launch as a SEPARATE process to avoid Pickle errors
        # We use sys.executable to make sure it uses your '.venv' python
        cmd = [
            sys.executable, 
            "train_wrapper.py", 
            model_pt, 
            json.dumps(task_data), 
            json.dumps(cleaned_params), 
            str(self.train_split_ratio.get())
        ]

        # Use Popen so the GUI doesn't freeze
        self.train_proc = subprocess.Popen(cmd)
        
        self.log(f"Training started (Standalone Mode). PID: {self.train_proc.pid}")
        self.train_btn.config(state="disabled", text="⌛ Training...")
        threading.Thread(target=self._monitor_training, daemon=True).start()

    def _monitor_training(self):
        """Checks if the subprocess is finished."""
        while self.train_proc and self.train_proc.poll() is None:
            time.sleep(1)
        self.root.after(0, self._finalize_training_state)

    def _finalize_training_state(self):
        """Resets buttons and logs when training is done or cancelled."""
        self.train_btn.config(state="normal", text="🔥 START TRAINING")
        self.train_proc = None
        self.log("Training engine has stopped/finished.")

    def stop_training(self):
        """Kills the standalone subprocess and all its GPU children."""
        if self.train_proc and self.train_proc.poll() is None:
            if messagebox.askyesno("Confirm", "Stop training?"):
                # On Windows, taskkill /F /T ensures all child GPU processes die too
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.train_proc.pid)])
                self.train_proc = None
                self._finalize_training_state()
                self.log("Training terminated.")
  
    def autolabel_current_view(self):
        if not self.image_list: return messagebox.showwarning("!", "No image loaded to label.")
        model_p = self.autolabel_model_path.get()
        if not model_p or not os.path.exists(model_p): return messagebox.showerror("Error", "Please select an Inference Model first.")

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
                
                # FIX: ONLY inject the real class name if it was ACTUALLY detected in this exact image
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

            # Save the newly discovered classes to the database
            if updated_classes:
                self.refresh_class_dropdown()
                selected = self.task_tree.selection()
                if selected:
                    idx = int(self.task_tree.item(selected[0])['values'][0])
                    self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                    self.save_json()

            self.redraw()
            self.log("Spot-labeling complete.")
        except Exception as e:
            self.log(f"Spot-labeling Error: {e}")
    # --- FILE MANAGMENT ---
    def load_recent_session(self):
        if hasattr(self, 'last_session'):
            self.training_session_path = self.last_session
            self.load_previews()
        else: messagebox.showinfo("!", "No recent labeling found.")

    def import_zip(self):
        paths = filedialog.askopenfilenames(filetypes=[("Zip files", "*.zip")])
        if not paths: return

        for p in paths:
            p = Path(p)
            ts = datetime.now().strftime("%H%M%S_%f")[:10] 
            short_folder = f"i_{ts}" 
            extract_to = Path(self.export_dir.get()) / short_folder
            os.makedirs(extract_to, exist_ok=True)
            
            self.log(f"Extracting {p.name} to {short_folder}...")
            try:
                # 1. Extract everything blindly
                with zipfile.ZipFile(p, 'r') as z:
                    z.extractall(extract_to)
                
                # 2. Setup the target pristine folder
                clean_data_dir = extract_to / "obj_train_data"
                os.makedirs(clean_data_dir, exist_ok=True)
                
                # 3. Hunt down every image no matter how deeply CVAT nested it
                all_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    all_images.extend(list(extract_to.rglob(ext)))
                
                if not all_images:
                    self.log(f"No images found in {p.name}")
                    shutil.rmtree(extract_to, ignore_errors=True)
                    continue

                # 4. Move all images and their matching .txt labels to the clean root folder
                for img_path in all_images:
                    if img_path.parent == clean_data_dir:
                        continue # Skip if already in the right spot
                        
                    # Move image
                    dest_img = clean_data_dir / img_path.name
                    if not dest_img.exists():
                        shutil.move(str(img_path), str(dest_img))
                    
                    # Check for matching label and move it too
                    txt_path = img_path.with_suffix('.txt')
                    if txt_path.exists():
                        dest_txt = clean_data_dir / txt_path.name
                        if not dest_txt.exists():
                            shutil.move(str(txt_path), str(dest_txt))

                # 5. Hunt down obj.names (CVAT's class list) anywhere in the zip
                obj_names_files = list(extract_to.rglob("obj.names"))
                task_classes = {0: "object"}
                if obj_names_files:
                    with open(obj_names_files[0], 'r') as f:
                        task_classes = {i: line.strip() for i, line in enumerate(f) if line.strip()}
                
                # 6. Aggressive Cleanup: Delete all messy nested folders left behind by CVAT
                for item in extract_to.iterdir():
                    if item.is_dir() and item.name != "obj_train_data":
                        shutil.rmtree(item, ignore_errors=True)
                    elif item.is_file():
                        os.remove(item) # Deletes loose junk files like train.txt

                # 7. Save the perfectly clean, flattened task to the database
                task_name = p.stem 
                try: 
                    rel_path = extract_to.relative_to(Path(__file__).parent)
                except ValueError: 
                    rel_path = extract_to

                self.save_task_to_db(task_name, rel_path, "import", classes=task_classes)
                self.log(f"Imported: {task_name} with {len(task_classes)} classes (Flattened successfully!).")

            except Exception as e:
                self.log(f"Import Failed: {e}")
        
        self.refresh_task_list()

    def load_previews(self):
        """Loads images from any folder starting with 'obj_train_data'."""
        if not self.training_session_path.exists():
            self.log("Error: Session path not found.")
            return

        # Find folders like obj_train_data_part9, obj_train_data, etc.
        data_folders = [d for d in self.training_session_path.glob("obj_train_data*") if d.is_dir()]
        
        if not data_folders:
            self.image_list = []
            self.log("No data folders found.")
            return

        # Combine images from all parts found (or just the first one)
        self.image_list = []
        for folder in data_folders:
            self.image_list.extend(sorted(list(folder.glob("*.jpg"))))

        if self.image_list:
            self.current_img_idx = 0
            self.show_image()
            self.log(f"Loaded {len(self.image_list)} images for editing.")
        else:
            self.log("No .jpg images found in data folders.")

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
        txt_p = img_p.with_suffix('.txt')
        updated_classes = False
        
        if txt_p.exists():
            with open(txt_p, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        c = int(parts[0])
                        # If a class number exists in the txt file but not in our dictionary, add it
                        if c not in self.class_names:
                            self.class_names[c] = f"class_{c}"
                            updated_classes = True
                    
                    c, cx, cy, nw, nh = map(float, line.split())
                    x1, y1 = (cx-nw/2)*self.canvas_w, (cy-nh/2)*self.canvas_h
                    x2, y2 = (cx+nw/2)*self.canvas_w, (cy+nh/2)*self.canvas_h
                    self.boxes.append([c, x1, y1, x2, y2])
                    
        # Save the automatically generated classes to the database
        if updated_classes:
            self.refresh_class_dropdown()
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                self.save_json()

        self.redraw()
        self.is_dirty = False
        self.idx_lbl.config(text=f"{self.current_img_idx+1}/{len(self.image_list)}")

    def next_img(self): 
        if self.current_img_idx < len(self.image_list)-1: 
            self.save_boxes() # AUTO-SAVE
            self.current_img_idx += 1; self.show_image()
    
    def prev_img(self): 
        if self.current_img_idx > 0: 
            self.save_boxes() # AUTO-SAVE
            self.current_img_idx -= 1; self.show_image()
    
    def browse_model(self):
        p = filedialog.askopenfilename()
        if p:
            self.model_path.set(p)
            self.log(f"Base training model set to: {Path(p).name}")
            m = YOLO(p)
            # Copy names directly from the model and refresh UI
            self.class_names = m.names.copy()
            
            selected = self.task_tree.selection()
            if selected:
                idx = int(self.task_tree.item(selected[0])['values'][0])
                self.tasks[idx]["classes"] = {str(k): v for k, v in self.class_names.items()}
                self.save_json()
                
            self.refresh_class_dropdown()
            self.log("Classes updated from loaded model.")

    def browse_autolabel_model(self):
            p = filedialog.askopenfilename()
            if p:
                self.autolabel_model_path.set(p)
                self.log(f"Inference model set to: {Path(p).name}")

    def browse_videos(self):
            """Appends new video selections to the existing list and updates the Listbox UI."""
            # Open file dialog to select multiple videos
            paths = filedialog.askopenfilenames(
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
            )
            
            if paths:
                for p in paths:
                    # Add to internal path list if not already there
                    if p not in self.video_files:
                        self.video_files.append(p)
                        # Add only the filename to the Listbox for a clean UI
                        self.v_listbox.insert(tk.END, Path(p).name)
                
                self.log(f"Added {len(paths)} video(s). Total in queue: {len(self.video_files)}")
    def log(self, msg):
        self.log_area.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"); self.log_area.see(tk.END)

    def save_snapshot(self):
        """Saves current boxes to stack before a change happens."""
        # We store a deep copy of the list
        self.undo_stack.append([list(box) for box in self.boxes])
        if len(self.undo_stack) > 20: # Limit memory
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