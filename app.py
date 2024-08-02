import os
import pandas as pd
import cv2
import json
import joblib
import base64
import numpy as np
from collections import deque
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection03,
    FaceAttributeTypeRecognition04,
)

# Set environment variable for QT platform
os.environ['QT_QPA_PLATFORM'] = 'xcb'

app = Flask(__name__, static_folder='../lms-react/build', static_url_path='/')
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin for WebSocket

def categorize_head_pose(pitch, yaw):
    if abs(yaw) < 10 and abs(pitch) < 10:
        return 'forward'
    elif pitch < -10:
        return 'down'
    elif yaw > 10:
        return 'right'
    elif yaw < -10:
        return 'left'
    else:
        return 'unknown'

# Load credentials from environment variables or a file
FACE_API_KEY = os.environ.get('FACE_API_KEY')
ENDPOINT_FACE = os.environ.get('ENDPOINT_FACE')

if not FACE_API_KEY or not ENDPOINT_FACE:
    credential = json.load(open("credential.json"))
    FACE_API_KEY = FACE_API_KEY or credential["FACE_API_KEY"]
    ENDPOINT_FACE = ENDPOINT_FACE or credential["ENDPOINT_FACE"]

# Initialize FaceClient
face_client = FaceClient(endpoint=ENDPOINT_FACE, credential=AzureKeyCredential(FACE_API_KEY))

# Load the pre-trained model and scaler
model = joblib.load('model4.pkl')
scaler = joblib.load('scaler3.pkl')

# Initialize a rolling window for inattentive frames
attention_window = deque(maxlen=10)  # 10-second window

# Flag to indicate whether the alert is active
alert_active = False

def extract_features(face):
    # Extract features from face attributes and landmarks
    features = {}

    if face.face_attributes:
        head_pose = face.face_attributes.head_pose
        features["pose"] = categorize_head_pose(head_pose.pitch, head_pose.yaw)
        features["pose_x"] = head_pose.yaw
        features["pose_y"] = head_pose.pitch

    if face.face_landmarks:
        landmarks = face.face_landmarks.as_dict()
        features["pupilLeft_x"] = landmarks["pupilLeft"]["x"]
        features["pupilLeft_y"] = landmarks["pupilLeft"]["y"]
        features["pupilRight_x"] = landmarks["pupilRight"]["x"]
        features["pupilRight_y"] = landmarks["pupilRight"]["y"]
        features["eyeLeftOuter_x"] = landmarks["eyeLeftOuter"]["x"]
        features["eyeLeftOuter_y"] = landmarks["eyeLeftOuter"]["y"]
        features["eyeLeftInner_x"] = landmarks["eyeLeftInner"]["x"]
        features["eyeLeftInner_y"] = landmarks["eyeLeftInner"]["y"]
        features["eyeRightOuter_x"] = landmarks["eyeRightOuter"]["x"]
        features["eyeRightOuter_y"] = landmarks["eyeRightOuter"]["y"]
        features["eyeRightInner_x"] = landmarks["eyeRightInner"]["x"]
        features["eyeRightInner_y"] = landmarks["eyeRightInner"]["y"]

    return features

def predict_attention(frame):
    # Read image content from the frame
    _, buffer = cv2.imencode('.jpg', frame)
    file_content = buffer.tobytes()

    # Detect faces
    result = face_client.detect(
        file_content,
        detection_model=FaceDetectionModel.DETECTION_03,
        recognition_model=FaceRecognitionModel.RECOGNITION_04,
        return_face_id=False,
        return_face_attributes=[
            FaceAttributeTypeDetection03.HEAD_POSE,
            FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION,
        ],
        return_face_landmarks=True,
        return_recognition_model=True,
        face_id_time_to_live=120,
    )

    if not result:
        return 0  # No face detected, assume not paying attention

    face = result[0]
    features = extract_features(face)

    # Convert features to DataFrame
    new_data = pd.DataFrame([features])

    # One-hot encode the 'pose' column
    new_data = pd.get_dummies(new_data, columns=['pose'])

    # Ensure all possible pose columns are present
    pose_columns = ['pose_down', 'pose_forward', 'pose_left', 'pose_right']
    for col in pose_columns:
        if col not in new_data.columns:
            new_data[col] = 0

    # Reindex the DataFrame to match the order of X_train columns
    X_train_columns = [
        'pose_x', 'pose_y', 'pose_down', 'pose_forward', 'pose_left', 'pose_right',
        'pupilLeft_x', 'pupilLeft_y', 'pupilRight_x', 'pupilRight_y',
        'eyeLeftOuter_x', 'eyeLeftOuter_y', 'eyeLeftInner_x', 'eyeLeftInner_y',
        'eyeRightOuter_x', 'eyeRightOuter_y', 'eyeRightInner_x', 'eyeRightInner_y'
    ]
    new_data = new_data.reindex(columns=X_train_columns, fill_value=0)

    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Predict the label for the new data
    predicted_label = model.predict(new_data_scaled)
    return predicted_label[0]


@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/courses')
def courses():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/course')
def course():
    return send_from_directory(app.static_folder, 'index.html')

# API endpoint to verify Flask is working
@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok"}), 200

@socketio.on('connect')
def handle_connect():
    emit('server_message', {'data': 'Hello from Flask'})

@socketio.on('image')
def handle_image(data):
    global attention_window
    global alert_active

    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    attention_status = predict_attention(frame)

    # Print the attention status for each frame
    attention_message = f'Attention Status: {"Attentive" if attention_status == 1 else "Not Attentive"}'
    emit('regular_status', {'status': attention_message})  # Emit the regular attention status

    # Update the rolling window
    attention_window.append(attention_status)

    # Check the number of inattentive frames in the last 10 seconds
    if not alert_active and attention_window.count(0) >= 8:  # If 8 or more out of the last 10 frames are inattentive
        alert_message = "We have noticed you are not watching the course. Please be attentive!"
        alert_active = True
        emit('attention_status', {'status': alert_message})

@socketio.on('acknowledge_alert')
def handle_acknowledge_alert():
    global alert_active
    global attention_window

    alert_active = False
    attention_window = deque(maxlen=10)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port)
