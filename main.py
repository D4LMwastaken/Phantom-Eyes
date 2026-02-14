from ultralytics import YOLO
import processing

print("Loading model...")
model = YOLO("models/yolo26n-pose.pt")

available_cameras = processing.list_available_cameras()

processing.process_webcam_threaded(model, 0,False) # 0 by default, 1 for ir if your computer has one