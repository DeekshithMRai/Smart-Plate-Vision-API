import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import requests

# Set the path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Try loading the YOLO model with error handling
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/display')
def display():
    data = []
    with open("car_plate_data.txt", "r") as file:
        lines = file.readlines()[1:]  # Skip the header line
        for line in lines:
            values = line.strip().split('\t')
            if len(values) == 3:
                number_plate, date, time = values
                response = get_vehicle_details(number_plate)
                data.append({
                    "NumberPlate": number_plate,
                    "Date": date,
                    "Time": time,
                    "Details": response
                })
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return render_template('display.html', data=data)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = "uploaded_video.mp4"
        file.save(file_path)
        process_video(file_path)
        return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    coco_path = "coco1.txt"

    if not os.path.exists(coco_path):
        print(f"File not found: {coco_path}")
        exit(1)

    with open(coco_path, "r") as my_file:
        class_list = my_file.read().split("\n")

    area = [(35, 375), (16, 456), (1015, 451), (965, 378)]

    count = 0
    processed_numbers = set()
    recent_detections = {}

    with open("car_plate_data.txt", "a") as file:
        if file.tell() == 0:
            file.write("NumberPlate\tDate\tTime\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        detections = results[0].boxes.data
        
        if detections is None:
            continue

        px = pd.DataFrame(detections).astype("float")

        for index, row in px.iterrows():
            x1, y1, x2, y2, _, class_idx = map(int, row[:6])
            class_name = class_list[class_idx]
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            
            if result >= 0:
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 10, 20, 20)

                try:
                    text = pytesseract.image_to_string(gray).strip()
                    text = text.replace(' ', '')  # Remove spaces
                    
                    if text and text not in processed_numbers:
                        current_datetime = datetime.now()
                        if text in recent_detections:
                            last_detected = recent_detections[text]
                            if (current_datetime - last_detected).total_seconds() < 60:
                                continue

                        processed_numbers.add(text)
                        recent_detections[text] = current_datetime
                        
                        with open("car_plate_data.txt", "a") as file:
                            file.write(f"{text}\t{current_datetime.date()}\t{current_datetime.time()}\n")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.imshow('crop', crop)
                except Exception as e:
                    print(f"OCR Error: {e}")

        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
        cv2.imshow("RGB", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def get_vehicle_details(number_plate):
    url = "http://127.0.0.1:5000/api/vehicle/details"
    payload = {"number_plate": number_plate}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    return response.json() if response.status_code == 200 else {"error": "Vehicle not found"}

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('', path)

if __name__ == "__main__":
    from app.api import api_bp  # Import the API blueprint

    app.register_blueprint(api_bp, url_prefix='/api')
    app.run(debug=True)
