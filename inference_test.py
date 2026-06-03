import cv2
import numpy as np
from ultralytics import YOLO
from hikrobot import CameraDevice, DeviceManager
from ctypes import cast, POINTER
from MvImport.MvCameraControl_class import MV_CC_DEVICE_INFO

def main():
    print("Loading YOLO model...")
    model = YOLO("E:\lampn\Programming\MechPartsDetection\AutoLabeller\MPD03062602.pt") 
    
    # --- TUNABLE INFERENCE PARAMETERS ---
    # Adjust these to optimize speed and accuracy for your mechanical parts
    yolo_params = {
        "imgsz": 640,          # Image size for inference (helps FPS if camera res is huge)
        "conf": 0.40,          # Minimum confidence threshold (0.0 to 1.0)
        "iou": 0.45,           # Intersection Over Union threshold for Non-Max Suppression
        "device": '0',         # '0' for primary GPU, 'cpu' for CPU processing
        "half": False,         # Set to True to use FP16 precision (speeds up GPU inference)
        "verbose": False,      # Set to False to stop YOLO from printing every single frame to terminal
        # "classes": [0, 2]    # Uncomment to ONLY detect specific class IDs
    }

    print("Scanning for connected Hikrobot cameras...")
    devices = DeviceManager.enumerate()

    # 1. Properly check if ANY cameras are detected using the C-struct attribute
    if devices.nDeviceNum == 0:
        print("Error: No cameras found. Check your connection and IP configuration.")
        return

    print(f"Success: Found {devices.nDeviceNum} camera(s).")

    # 2. Properly unpack the first camera from the C-pointer array
    device_info = cast(devices.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = CameraDevice(device_info)

    if not cam.create_handle_and_open():
        print("Error: Failed to open camera handle.")
        return

    print("Camera connected successfully. Starting video stream...")
    cam.start_grabbing()

    try:
        while True:
            frame_data = cam.get_one_frame(timeout_ms=5000)
            
            if frame_data:
                raw_bytes, info = frame_data
                rgb_bytes = cam.convert_to_rgb(raw_bytes, info)
                
                # Reshape to (Height, Width, Channels)
                img_rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((info.nHeight, info.nWidth, 3))
                
                # Convert to BGR for OpenCV / YOLO compatibility
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # --- RUN INFERENCE WITH KWARGS ---
                # We unpack our dictionary directly into the model call
                results = model(img_bgr, **yolo_params)
                
                current_frame_results = results[0] 

                # Process individual classes (Example)
                for box in current_frame_results.boxes:
                    class_id = int(box.cls[0])                     
                    class_name = model.names[class_id]             
                    confidence = float(box.conf[0])                
                    
                    # Uncomment if you want to print live detections
                    # print(f"Detected: '{class_name}' ({confidence * 100:.1f}%)")

                # Visualize and display
                annotated_frame = current_frame_results.plot()
                
                # Optional: Scale down the display window if the camera is e.g., 5MP or larger
                # annotated_frame = cv2.resize(annotated_frame, (1280, 720))

                cv2.imshow("Hikrobot - YOLO Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key pressed.")
                    break
            else:
                print("Warning: Frame grab timeout.")

    except KeyboardInterrupt:
        print("\nStream interrupted by user.")

    finally:
        print("Cleaning up resources...")
        cam.stop_grabbing()
        cam.close_and_destroy()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()