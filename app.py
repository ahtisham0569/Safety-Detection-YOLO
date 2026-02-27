from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import base64
import io
from PIL import Image

app = Flask(__name__)

model = YOLO('best.pt')

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def get_detections(image):
    results = model(image)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detections.append({
            'x': x1,
            'y': y1,
            'w': x2 - x1,
            'h': y2 - y1,
            'label': model.names[cls_id],
            'confidence': conf
        })
    return detections

@app.route('/')
def index():
    return "YOLO Safety Detection Running"

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        image = base64_to_image(data['image'])
        detections = get_detections(image)
        return jsonify({'detections': detections})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
