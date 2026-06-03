import os
import json
from pathlib import Path

# --- THE MASTER CLASS MAPPING ---
# This forces all tasks to obey this exact order
TARGET_MAPPING = {
    "screw": 0,
    "stamp": 1
}

def align_classes():
    db_path = "tasks_db.json"
    
    if not os.path.exists(db_path):
        print("Error: tasks_db.json not found in this folder.")
        return

    with open(db_path, "r") as f:
        tasks = json.load(f)

    changes_made = False

    for task in tasks:
        current_classes = task.get("classes", {})
        
        translation = {}
        needs_fixing = False
        new_classes = {}
        
        # Figure out which IDs are wrong in this task
        for old_id_str, class_name in current_classes.items():
            if class_name in TARGET_MAPPING:
                correct_id = TARGET_MAPPING[class_name]
                if int(old_id_str) != correct_id:
                    translation[old_id_str] = correct_id
                    needs_fixing = True
                new_classes[str(correct_id)] = class_name
            else:
                print(f"Warning: Unknown class '{class_name}' found. Add it to TARGET_MAPPING if you want to keep it.")
                
        # If wrong IDs are found, rewrite the .txt files
        if needs_fixing:
            print(f"Fixing mismatch in task: {task['name']}...")
            task_path = Path(task['path'])
            data_folder = task_path / "obj_train_data"
            
            if data_folder.exists():
                count = 0
                for txt_file in data_folder.glob("*.txt"):
                    with open(txt_file, "r") as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            old_id = parts[0]
                            # Swap the wrong ID for the correct ID
                            if old_id in translation:
                                parts[0] = str(translation[old_id])
                            new_lines.append(" ".join(parts) + "\n")
                            
                    with open(txt_file, "w") as f:
                        f.writelines(new_lines)
                    count += 1
                
                print(f"  -> Rewrote {count} label files.")
            
            # Update the JSON dictionary
            task["classes"] = new_classes
            changes_made = True
            print(f"  -> Updated JSON classes to {new_classes}")

    # Save the fixed Database
    if changes_made:
        with open(db_path, "w") as f:
            json.dump(tasks, f, indent=4)
        print("\nSUCCESS: All tasks are now perfectly aligned! You are ready to train.")
    else:
        print("\nNo mismatches found. Everything is already perfectly aligned.")

if __name__ == "__main__":
    align_classes()