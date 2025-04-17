import os
import cv2
import numpy as np
from face_utils import (
    STUDENT_IMAGES_FOLDER, 
    ENCODINGS_FILE, 
    get_face_analyzer,
    build_attention_model,
    enhance_image,
    save_encodings,
    batch_process_faces,
    normalize,
    is_good_quality_face
)

def create_student_embeddings():
    """Process all student images and create embeddings database"""
    # Initialize models
    face_analyzer = get_face_analyzer()
    attention_model = build_attention_model()
    attention_model.compile(optimizer='adam', loss='mse')
    
    # Arrays to store embeddings and names
    student_encodings = []
    student_names = []
    
    # Process each image in the student images folder
    print(f"Processing student images from: {STUDENT_IMAGES_FOLDER}")
    
    # Collect images and names first
    image_paths = []
    names = []
    
    for filename in os.listdir(STUDENT_IMAGES_FOLDER):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(STUDENT_IMAGES_FOLDER, filename)
            student_name = os.path.splitext(filename)[0]  # Use filename (without extension) as student name
            
            image_paths.append(image_path)
            names.append(student_name)
    
    print(f"Found {len(image_paths)} student images to process")
    
    # Process images in batches
    batch_size = 10  # Process 10 images at a time
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_names = names[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1} with {len(batch_paths)} images")
        
        # Process each image in the batch
        all_faces = []
        valid_indices = []
        
        for j, (path, name) in enumerate(zip(batch_paths, batch_names)):
            print(f"Reading image: {os.path.basename(path)}")
            
            # Read and enhance image
            img = cv2.imread(path)
            if img is None:
                print(f"ERROR: Unable to read {path}")
                continue
                
            # Enhance image for low-light conditions
            img = enhance_image(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get faces from the image
            faces = face_analyzer.get(img_rgb)
            if len(faces) > 0:
                face = faces[0]
                
                # Check if face is of good quality for registration
                if not is_good_quality_face(face):
                    print(f"Low quality face in {os.path.basename(path)}: incorrect angle or pose, skipping")
                    continue
                
                # Get bounding box for size validation
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                
                # Extract face crop to check dimensions
                face_img = img_rgb[y:y2, x:x2]
                
                # Validate face crop dimensions
                if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                    print(f"Face crop too small in {os.path.basename(path)}: {face_img.shape}, skipping")
                    continue
                
                all_faces.append(face)
                valid_indices.append(j)
            else:
                print(f"No face detected in {path}")
        
        if all_faces:
            # Process all valid faces in batch
            batch_embeddings = batch_process_faces(all_faces, attention_model)
            
            # Add embeddings and names to our lists
            for idx, embedding in zip(valid_indices, batch_embeddings):
                student_encodings.append(embedding)
                student_names.append(batch_names[idx])
                print(f"Successfully processed {os.path.basename(batch_paths[idx])}")
    
    # Save embeddings to pickle file
    if student_encodings:
        save_encodings(ENCODINGS_FILE, student_encodings, student_names)
        print(f"Successfully created embeddings for {len(student_encodings)} students")
    else:
        print("No student embeddings were created. Check the images folder.")

if __name__ == "__main__":
    create_student_embeddings() 