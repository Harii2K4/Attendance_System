
import sys
import os
import json
import numpy as np
import cv2
import tensorflow as tf
import pickle
from datetime import datetime

# Add the parent directory to path so we can import face_utils
sys.path.append('C:\\Users\\harik\\OneDrive\\Desktop\\Attendace_System_Machine_Vision\\Attendance_System')
from modified.face_utils import get_face_analyzer, build_attention_model, enhance_image, normalize, cosine_similarity

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_face_encodings(image_path, output_path):
    """Extract face encodings from an image and save to a JSON file"""
    print(f"[{datetime.now()}] Processing image: {image_path}")
    
    # Initialize models
    face_analyzer = get_face_analyzer()
    attention_model = build_attention_model()
    attention_model.compile(optimizer='adam', loss='mse')
    
    # Read and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[{datetime.now()}] ERROR: Could not read image {image_path}")
        with open(output_path, 'w') as f:
            json.dump({"encodings": [], "success": False, "error": "Could not read image"}, f)
        return
    
    # Enhance image for low-light conditions
    img = enhance_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get faces from the image
    faces = face_analyzer.get(img_rgb)
    print(f"[{datetime.now()}] Found {len(faces)} faces in image")
    
    encodings = []
    
    if len(faces) > 0:
        for face in faces:
            print(f"[{datetime.now()}] Processing face...")
            # Apply attention model to the embedding
            embedding = attention_model.predict(face.embedding[np.newaxis, :])[0]
            embedding = normalize(embedding)
            encodings.append(embedding)
    
    print(f"[{datetime.now()}] Generated {len(encodings)} face encodings")
    with open(output_path, 'w') as f:
        json.dump({"encodings": encodings, "success": True}, f, cls=NumpyEncoder)
    print(f"[{datetime.now()}] Saved encodings to {output_path}")

def match_with_anonymous(face_encoding, anonymous_file, output_path):
    """Check if a face matches any anonymous face in the database"""
    print(f"[{datetime.now()}] Checking for matching anonymous faces in {anonymous_file}")
    
    if not os.path.exists(anonymous_file):
        print(f"[{datetime.now()}] Anonymous file does not exist")
        with open(output_path, 'w') as f:
            json.dump({"matched": False}, f)
        return
    
    # Load anonymous encodings
    with open(anonymous_file, 'rb') as f:
        data = pickle.load(f)
    
    if not data or 'encodings' not in data or 'names' not in data:
        print(f"[{datetime.now()}] Invalid anonymous data format")
        with open(output_path, 'w') as f:
            json.dump({"matched": False}, f)
        return
    
    # Convert the face encoding from list to numpy array
    if isinstance(face_encoding, list):
        face_encoding = np.array(face_encoding)
    
    matched_index = -1
    matched_name = None
    best_similarity = 0.75  # Threshold for matching
    
    # Compare with each anonymous face
    for i, anon_encoding in enumerate(data['encodings']):
        similarity = cosine_similarity(face_encoding, anon_encoding)
        print(f"[{datetime.now()}] Similarity with {data['names'][i]}: {similarity}")
        
        if similarity > best_similarity:
            best_similarity = similarity
            matched_index = i
            matched_name = data['names'][i]
    
    result = {
        "matched": matched_index >= 0,
        "index": matched_index,
        "name": matched_name,
        "similarity": float(best_similarity) if matched_index >= 0 else 0.0
    }
    
    print(f"[{datetime.now()}] Match result: {result}")
    with open(output_path, 'w') as f:
        json.dump(result, f)

def update_anonymous_database(anonymous_file, index_to_remove, output_path):
    """Remove an entry from the anonymous database"""
    print(f"[{datetime.now()}] Updating anonymous database: removing index {index_to_remove}")
    
    if not os.path.exists(anonymous_file):
        print(f"[{datetime.now()}] Anonymous file does not exist")
        with open(output_path, 'w') as f:
            json.dump({"success": False, "error": "File not found"}, f)
        return
    
    # Load anonymous encodings
    with open(anonymous_file, 'rb') as f:
        data = pickle.load(f)
    
    if not data or 'encodings' not in data or 'names' not in data:
        print(f"[{datetime.now()}] Invalid anonymous data format")
        with open(output_path, 'w') as f:
            json.dump({"success": False, "error": "Invalid data format"}, f)
        return
    
    # Make sure the index is valid
    if index_to_remove < 0 or index_to_remove >= len(data['encodings']):
        print(f"[{datetime.now()}] Invalid index: {index_to_remove}")
        with open(output_path, 'w') as f:
            json.dump({"success": False, "error": "Invalid index"}, f)
        return
    
    # Remove the entry
    removed_name = data['names'][index_to_remove]
    data['encodings'].pop(index_to_remove)
    data['names'].pop(index_to_remove)
    
    # Save the updated database
    with open(anonymous_file, 'wb') as f:
        pickle.dump(data, f)
    
    result = {
        "success": True,
        "removed_name": removed_name,
        "remaining_count": len(data['names'])
    }
    
    print(f"[{datetime.now()}] Anonymous database updated: {result}")
    with open(output_path, 'w') as f:
        json.dump(result, f)

def update_student_encodings(encodings_file, student_id, student_name, face_encoding, photo_path, output_path):
    """Add or update a student in the encodings database"""
    print(f"[{datetime.now()}] Updating student encodings for {student_name} (ID: {student_id})")
    
    # Initialize data structure
    data = {"students": []}
    
    # Load existing data if file exists
    if os.path.exists(encodings_file):
        try:
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
            if not data or 'students' not in data:
                data = {"students": []}
        except Exception as e:
            print(f"[{datetime.now()}] Error loading encodings file: {e}")
            data = {"students": []}
    
    # Convert the face encoding from list to numpy array if needed
    if isinstance(face_encoding, list):
        face_encoding = np.array(face_encoding)
    
    # Check if student already exists
    student_exists = False
    for i, student in enumerate(data['students']):
        if student.get('id') == student_id or student.get('name') == student_name:
            # Update existing student
            data['students'][i]['name'] = student_name
            data['students'][i]['id'] = student_id
            data['students'][i]['encodings'] = [face_encoding]
            data['students'][i]['photoPath'] = photo_path
            student_exists = True
            break
    
    # Add new student if not found
    if not student_exists:
        data['students'].append({
            'id': student_id,
            'name': student_name,
            'encodings': [face_encoding],
            'photoPath': photo_path
        })
    
    # Save the updated database
    with open(encodings_file, 'wb') as f:
        pickle.dump(data, f)
    
    result = {
        "success": True,
        "id": student_id,
        "name": student_name,
        "action": "updated" if student_exists else "added"
    }
    
    print(f"[{datetime.now()}] Student encodings updated: {result}")
    with open(output_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    command = sys.argv[1]
    
    if command == "encode":
        # Extract face encodings from an image
        # Args: encode image_path output_path
        image_path = sys.argv[2]
        output_path = sys.argv[3]
        get_face_encodings(image_path, output_path)
    
    elif command == "match_anonymous":
        # Match a face with anonymous database
        # Args: match_anonymous face_encoding_json anonymous_file output_path
        face_encoding_json = sys.argv[2]
        anonymous_file = sys.argv[3]
        output_path = sys.argv[4]
        
        # Load the face encoding
        with open(face_encoding_json, 'r') as f:
            data = json.load(f)
        
        if data and 'encodings' in data and len(data['encodings']) > 0:
            match_with_anonymous(data['encodings'][0], anonymous_file, output_path)
        else:
            with open(output_path, 'w') as f:
                json.dump({"matched": False, "error": "No valid face encoding"}, f)
    
    elif command == "update_anonymous":
        # Update anonymous database by removing an entry
        # Args: update_anonymous anonymous_file index_to_remove output_path
        anonymous_file = sys.argv[2]
        index_to_remove = int(sys.argv[3])
        output_path = sys.argv[4]
        update_anonymous_database(anonymous_file, index_to_remove, output_path)
    
    elif command == "update_student":
        # Add or update a student in the encodings database
        # Args: update_student encodings_file student_id student_name face_encoding_json photo_path output_path
        encodings_file = sys.argv[2]
        student_id = sys.argv[3]
        student_name = sys.argv[4]
        face_encoding_json = sys.argv[5]
        photo_path = sys.argv[6]
        output_path = sys.argv[7]
        
        # Load the face encoding
        with open(face_encoding_json, 'r') as f:
            data = json.load(f)
        
        if data and 'encodings' in data and len(data['encodings']) > 0:
            update_student_encodings(encodings_file, student_id, student_name, data['encodings'][0], photo_path, output_path)
        else:
            with open(output_path, 'w') as f:
                json.dump({"success": False, "error": "No valid face encoding"}, f)
    
    else:
        print(f"[{datetime.now()}] Unknown command: {command}")
        sys.exit(1)
