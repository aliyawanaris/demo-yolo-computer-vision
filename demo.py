import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# pyrefly: ignore [missing-import]
import cv2
import time
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import draw_boxes as Boxes
# pyrefly: ignore [missing-import]
import numpy as np

def run_demo(source=0, model_path='yolov8s.pt'):
    # Load model
    print(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        print("Falling back to yolov8n.pt (Nano version)...")
        model = YOLO('yolov8n.pt')

    # Initialize video capture
    # source can be 0 (webcam), a filename, or a URL (IP camera)
    vid = cv2.VideoCapture(source)
    
    if not vid.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Demo started. Press 'q' to quit.")
    
    # Get original classes from model if needed
    model_classes = model.names
    
    while True:
        ret, img = vid.read()
        if not ret:
            print("Finished or could not read frame.")
            break
            
        start = time.time()
        # Predict
        results = model.predict(img, conf=0.25, verbose=False)
        end = time.time()
        
        fps = 1.0 / (end - start)
        
        # Draw results
        # Note: Boxes.drawBoxes expects a specific format: [x1, y1, x2, y2, score, class]
        # ultralytics results[0].boxes.data is exactly that
        Boxes.drawBoxes(img, results[0].boxes.data, labels=model_classes, score=True)
        
        # Add FPS text
        cv2.putText(img, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow('Traffic Detection Demo', img)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8 Traffic Detection Demo')
    parser.add_argument('--source', type=str, default='0', help='0 for webcam, filename for video, or http://IP:PORT/video for smartphone')
    parser.add_argument('--model', type=str, default='best_m.pt', help='Path to model weights')
    
    args = parser.parse_args()
    
    # Handle numeric source (webcam index)
    source = args.source
    if source.isdigit():
        source = int(source)
        
    run_demo(source=source, model_path=args.model)
