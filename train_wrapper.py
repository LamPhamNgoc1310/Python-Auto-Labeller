# train_wrapper.py
import sys
import json
import TrainModel

if __name__ == "__main__":
    # Receive arguments from the GUI
    model_pt = sys.argv[1]
    task_data_json = sys.argv[2] # We pass the list of (path, weight) as JSON
    params_json = sys.argv[3]
    ratio = float(sys.argv[4])

    # Decode the JSON data
    task_data = json.loads(task_data_json)
    params = json.loads(params_json)

    # Start the real training logic
    TrainModel.run_training_process(model_pt, task_data, params, ratio)