
import sys
import os
import json
import numpy as np
import cv2
import tensorflow as tf

# Add the correct path to the modified directory
sys.path.append('C:\\Users\\harik\\OneDrive\\Desktop\\Attendace_System_Machine_Vision\\Attendance_System')
from modified.face_utils import get_face_analyzer, build_attention_model, enhance_image, normalize

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_encodings(image_path, json_path):
    # Initialize models from face_utils
    face_analyzer = get_face_analyzer()
    attention_model = build_attention_model()
    attention_model.compile(optimizer='adam', loss='mse')
    
    # Read and process image
    img = cv2.imread(image_path)
    if img is None:
        with open(json_path, 'w') as f:
            json.dump({"encodings": []}, f)
        return
        
    # Enhance image for low-light conditions
    img = enhance_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get faces from the image
    faces = face_analyzer.get(img_rgb)
    
    encodings = []
    
    if len(faces) > 0:
        for face in faces:
            # Apply attention model to the embedding
            embedding = attention_model.predict(face.embedding[np.newaxis, :])[0]
            embedding = normalize(embedding)
            encodings.append(embedding)
    
    with open(json_path, 'w') as f:
        json.dump({"encodings": encodings}, f, cls=NumpyEncoder)

if __name__ == "__main__":
    image_path = sys.argv[1]
    json_path = sys.argv[2]
    get_encodings(image_path, json_path)
    