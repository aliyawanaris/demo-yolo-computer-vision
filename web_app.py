import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen' # Disable GUI for web server

from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import draw_boxes as Boxes
import numpy as np

app = Flask(__name__)

# Load Model
model = YOLO('yolov8n.pt')

def gen_frames():
    # Gunakan webcam (0)
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # YOLO Inference
            results = model.predict(frame, conf=0.25, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                
                for box, cls, score in zip(boxes, classes, scores):
                    label = f"{model.names[int(cls)]} {score:.2f}"
                    Boxes.drawBox(frame, box, label=label, color=(0, 255, 0))

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>YOLOv8 Web Stream</title>
        <style>
          body { font-family: sans-serif; text-align: center; background: #121212; color: white; }
          img { border: 5px solid #333; border-radius: 10px; max-width: 90%; }
        </style>
      </head>
      <body>
        <h1>Traffic Detection - Live Stream</h1>
        <img src="/video_feed">
        <p>Akses alamat ini dari browser HP Anda untuk melihat hasil deteksi.</p>
      </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Ganti 0.0.0.0 agar bisa diakses dari perangkat lain di Wi-Fi yang sama
    app.run(host='0.0.0.0', port=5000)
