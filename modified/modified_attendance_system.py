import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from face_utils import (
    STUDENT_IMAGES_FOLDER,
    ENCODINGS_FILE,
    ANONYMOUS_FILE,
    ANONYMOUS_FOLDER,
    ATTENDANCE_FILE,
    THRESHOLD,
    get_face_analyzer,
    build_attention_model,
    normalize,
    enhance_image,
    detect_backlighting,
    is_blurry,
    load_encodings,
    save_encodings,
    euclidean_distance,
    add_padding_to_bbox,
    batch_process_faces,
    is_good_quality_face,
    detect_mask_or_occlusion,
    cosine_similarity,
    adaptive_threshold,
    FaceTracker,
    find_best_match
)

# Array definitions
known_encodings = []
known_names = []
anonymous_encodings = []
anonymous_names = []

# Instantiate the attention model (SE block)
attention_model = build_attention_model()
attention_model.compile(optimizer='adam', loss='mse')

def create_encodings(folder_path):
    # Initialize InsightFace model for creating the encodings
    face_analyzer = get_face_analyzer()
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder_path, filename)
            name = os.path.splitext(filename)[0]  # Use filename (without extension) as label
            
            # Read and process image
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
                # Use the first detected face's embedding
                face = faces[0]
                # Apply attention model to the embedding
                embedding = attention_model.predict(face.embedding[np.newaxis, :])[0]
                embedding = normalize(embedding)
                
                known_encodings.append(embedding)
                known_names.append(name)
            else:
                print(f"No face detected in {path}")
                continue

# Add a new function to check if anonymous face is unique
def is_unique_anonymous_face(embedding, existing_encodings, threshold=0.8):
    """
    Check if a face embedding is sufficiently different from existing encodings
    
    Args:
        embedding: New face embedding to check
        existing_encodings: List of existing face embeddings to compare against
        threshold: Similarity threshold (higher means more strict matching)
        
    Returns:
        bool: True if this face is unique (not similar to any existing faces)
    """
    if not existing_encodings:
        return True
        
    # Calculate similarity with all existing anonymous encodings
    for existing_embedding in existing_encodings:
        similarity = cosine_similarity(embedding, existing_embedding)
        # If similarity is above threshold, this face is similar to an existing one
        if similarity > threshold:
            return False
            
    # If no matches found, this face is unique
    return True

def attendance_system(start_time):
    global df
    global anonymous_encodings, anonymous_names
    
    # Keep track of which anonymous people are already in the Excel sheet
    anonymous_in_excel = []
    
    # Track anonymous embeddings that are in the Excel sheet
    anonymous_excel_encodings = []

    # Load the InsightFace face analyzer
    face_analyzer = get_face_analyzer()
    
    # Initialize face tracker for temporal consistency
    face_tracker = FaceTracker(max_history=20, similarity_threshold=0.65)  # Maintain larger history for better tracking
    
    # Frame counter for tracking
    frame_counter = 0
    
    # Map of track IDs to recognized names
    track_names = {}
    
    # Confidence counters for each track
    track_confidence = {}
    
    # Blur detection counter - tracks consecutive blurry frames
    blur_counter = 0
    
    # Backlight detection counter
    backlight_counter = 0
    
    # Track consecutive occlusion frames for the same person
    occlusion_consistency_counter = {}
    
    # Store original embeddings for partially occluded faces
    last_clear_face_embeddings = {}

    # Start webcam
    video_capture = cv2.VideoCapture(0)
    # Counter for anonymous faces
    anonymous_counter = len(anonymous_encodings) + 1

    # Check which anonymous names are already in Excel
    if 'Name' in df.columns:
        for name in df['Name']:
            if isinstance(name, str) and name.startswith("Anonymous_"):
                anonymous_in_excel.append(name)
                # Find the corresponding encoding
                if name in anonymous_names:
                    idx = anonymous_names.index(name)
                    anonymous_excel_encodings.append(anonymous_encodings[idx])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
            
        # Increment frame counter
        frame_counter += 1
        
        # Disabled blur detection
        blur_counter = 0
        frame_is_blurry = False
        
        # Check for backlighting and apply enhanced correction
        frame_is_backlit = detect_backlighting(frame)
        if frame_is_backlit:
            backlight_counter += 1
            if backlight_counter > 3:  # Apply enhanced correction after confirming backlighting
                cv2.putText(frame, "BACKLIT - ENHANCING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                # Apply enhanced image correction specifically for backlit images
                frame = enhance_image(frame)
        else:
            backlight_counter = 0
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using InsightFace
        faces = face_analyzer.get(rgb_frame)
        
        if not faces:
            # Display the frame with no faces
            cv2.imshow("Face Recognition Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # Process all face embeddings in batch
        embeddings = batch_process_faces(faces, attention_model)
        
        # Process each face with corresponding embedding
        for face, embedding in zip(faces, embeddings):
            # Skip processing was removed - always process all faces
                
            # Extract face location
            bbox = face.bbox.astype(int)
            x, y, x2, y2 = bbox
            w, h = x2 - x, y2 - y
            
            # Add padding to the bounding box
            x1, y1, x2, y2 = add_padding_to_bbox(bbox, rgb_frame.shape)
            face_img = rgb_frame[y1:y2, x1:x2]
            
            # Store the image in the face object for occlusion detection
            face._img = face_img.copy()
            
            # Validate face crop dimensions
            if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                print(f"Face crop too small: {face_img.shape}, skipping")
                continue
                
            # Removed blur check for face image
                
            # Check for face occlusion (mask)
            is_occluded, occlusion_type, occlusion_score = detect_mask_or_occlusion(face)
            if is_occluded:
                print(f"Detected {occlusion_type} with score {occlusion_score:.2f}")
                # Use more lenient threshold for masked faces
                base_threshold = 0.65  # Lower threshold for masked faces
            else:
                base_threshold = 0.75  # Normal threshold for clear faces
            
            # For backlit images, adjust thresholds even further
            if frame_is_backlit:
                base_threshold *= 0.9  # Make threshold more lenient for backlit images
                
            # Get track ID for temporal consistency - pass the face object for occlusion detection
            track_id = face_tracker.get_track_id(embedding, frame_counter, face=face)
            
            # Find the best match among known faces
            name, score, is_match = find_best_match(
                embedding, 
                known_encodings, 
                known_names, 
                face=face, 
                use_cosine=True,  # Use cosine similarity instead of Euclidean
                base_threshold=base_threshold * 0.85  # Lower threshold for all face comparisons
            )
            
            # Apply temporal consistency to the recognition
            if track_id in track_names:
                # We've seen this face before
                prev_name = track_names[track_id]
                
                # Initialize occlusion counter if not present
                if track_id not in occlusion_consistency_counter:
                    occlusion_consistency_counter[track_id] = 0
                
                # If face is occluded, maintain previous identity more strongly
                if is_occluded:
                    occlusion_consistency_counter[track_id] += 1
                    
                    # Store occlusion score to use in decision making
                    occlusion_weight = min(occlusion_score, 0.9)  # Cap at 0.9 to prevent complete override
                    
                    # If this face has been consistently occluded, favor previous identification
                    # The higher the occlusion, the more we favor previous identification
                    occlusion_threshold = 2
                    if occlusion_score > 0.7:  # Heavy occlusion
                        occlusion_threshold = 1  # Require fewer frames to maintain identity
                        
                    if occlusion_consistency_counter[track_id] >= occlusion_threshold:
                        # Only stick with previous name if it's a named person (not Unregistered)
                        if prev_name != "Unregistered" and not prev_name.startswith("Unknown"):
                            # For heavily occluded faces, always keep previous identity
                            if occlusion_score > 0.5:
                                name = prev_name
                                is_match = True
                                print(f"Maintaining identity {prev_name} through occlusion (score: {occlusion_score:.2f})")
                    
                    # If we have a clear face embedding stored for this track, compare with it
                    if track_id in last_clear_face_embeddings:
                        clear_embedding = last_clear_face_embeddings[track_id]
                        similarity = cosine_similarity(embedding, clear_embedding)
                        
                        # If similar enough to the clear face, keep the identity
                        # Use adaptive threshold based on occlusion level
                        clear_face_threshold = 0.65 - (occlusion_weight * 0.25)
                        if similarity > clear_face_threshold:
                            name = prev_name
                            is_match = True
                            print(f"Matched occluded face with stored clear face: {similarity:.2f} > {clear_face_threshold:.2f}")
                else:
                    # Reset occlusion counter when face is clearly visible
                    occlusion_consistency_counter[track_id] = 0
                    
                    # Store clear face embedding for future comparison during occlusion
                    if not is_occluded or occlusion_score < 0.3:
                        last_clear_face_embeddings[track_id] = embedding.copy()
                
                if prev_name == name:
                    # Same identity detected again, increase confidence more aggressively
                    if track_id not in track_confidence:
                        track_confidence[track_id] = 2  # Start with higher confidence
                    else:
                        track_confidence[track_id] += 2  # Increase by 2 instead of 1
                        
                    # Cap confidence at higher value
                    track_confidence[track_id] = min(track_confidence[track_id], 10)
                else:
                    # Different identity detected, decrease confidence more slowly
                    if track_id not in track_confidence:
                        track_confidence[track_id] = 0
                    else:
                        track_confidence[track_id] -= 0.5  # Decrease by 0.5 instead of 1
                    
                    # If confidence goes below zero, update the identity
                    if track_confidence[track_id] < 0:
                        track_names[track_id] = name
                        track_confidence[track_id] = 1
            else:
                # First time seeing this face
                track_names[track_id] = name
                track_confidence[track_id] = 2  # Start with higher initial confidence
                occlusion_consistency_counter[track_id] = 0
            
            # Update the track with this name for consistency tracking
            face_tracker.update_track_name(track_id, name)
            
            # Get the most consistent name for this track
            consistent_name = face_tracker.get_consistent_name(track_id, name)
            
            # Use the consistent name for display and attendance
            name = consistent_name
            
            # Also update our track_names dictionary
            track_names[track_id] = name
            
            # Flag to track if we should add this person to Excel
            should_add_to_excel = True
            
            # If person is not identified as a student, check anonymous faces
            if not is_match:
                found_anonymous = False
                
                # Find the best match among anonymous faces using cosine similarity
                anon_name, anon_score, anon_match = find_best_match(
                    embedding, 
                    anonymous_encodings, 
                    anonymous_names, 
                    face=face, 
                    use_cosine=True,
                    base_threshold=base_threshold * 0.93  # Slightly lower threshold for anonymous faces
                )
                
                if anon_match:
                    name = anon_name
                    found_anonymous = True
                    
                    # Check if this anonymous person is already in Excel
                    if name in anonymous_in_excel:
                        should_add_to_excel = False
                
                # Register new anonymous face if not found and is good quality
                if not found_anonymous:
                    # Removed blur check for anonymous registration
                    
                    # Don't register anonymous faces from backlit frames without multiple confirmations
                    if frame_is_backlit and backlight_counter < 5:
                        name = "Unregistered"
                        should_add_to_excel = False
                        print("Not registering anonymous face from backlit frame without confirmation")
                        continue
                        
                    # Don't require perfect face quality for anonymous registration if masked
                    should_register = is_good_quality_face(face)
                    
                    # IMPORTANT: Never register anonymous faces with any occlusion
                    # This prevents creating new anonymous identities when face is partially covered
                    if is_occluded:
                        # Don't register any anonymous faces when occlusion is detected
                        should_register = False
                        print(f"Occlusion detected ({occlusion_score:.2f}), not registering new anonymous face")
                        
                        # If track ID has a stored clear face, try to match with existing anonymous faces
                        if track_id in last_clear_face_embeddings:
                            clear_embedding = last_clear_face_embeddings[track_id]
                            best_anon_match = None
                            best_anon_similarity = 0.7  # Minimum similarity threshold
                            
                            # Search for best match among anonymous faces
                            for i, anon_embedding in enumerate(anonymous_encodings):
                                similarity = cosine_similarity(clear_embedding, anon_embedding)
                                if similarity > best_anon_similarity:
                                    best_anon_similarity = similarity
                                    best_anon_match = anonymous_names[i]
                            
                            # If found a good match with stored clear face, use that identity
                            if best_anon_match:
                                name = best_anon_match
                                print(f"Matched occluded face with existing anonymous person: {name} ({best_anon_similarity:.2f})")
                    
                    if should_register:
                        # Use a higher similarity threshold (0.95) to reduce duplicate anonymous entries
                        if is_unique_anonymous_face(embedding, anonymous_excel_encodings, threshold=0.95):
                            print(f"New anonymous face detected")
                            # Save enhanced version of the face
                            enhanced_face_img = enhance_image(cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                            face_filename = os.path.join(ANONYMOUS_FOLDER, f"Anonymous_{anonymous_counter}.jpg")
                            cv2.imwrite(face_filename, enhanced_face_img)
                            
                            anonymous_encodings.append(embedding)
                            anon_name = f"Anonymous_{anonymous_counter}"
                            anonymous_names.append(anon_name)
                            save_encodings(ANONYMOUS_FILE, anonymous_encodings, anonymous_names)
                            name = anon_name
                            anonymous_counter += 1
                            
                            # Add to our tracking lists
                            anonymous_excel_encodings.append(embedding)
                            anonymous_in_excel.append(name)
                            
                            # Reset occlusion counter for this new identity
                            occlusion_consistency_counter[track_id] = 0
                        else:
                            print("Similar anonymous face already exists in Excel, not adding a new entry")
                            should_add_to_excel = False
                            # Find the most similar existing anonymous face
                            best_similarity = -1
                            best_match_name = None
                            for i, existing_embedding in enumerate(anonymous_excel_encodings):
                                similarity = cosine_similarity(embedding, existing_embedding)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    idx = anonymous_excel_encodings.index(existing_embedding)
                                    best_match_name = anonymous_in_excel[i]
                            
                            if best_match_name:
                                name = best_match_name
                                print(f"Matched with existing anonymous entry: {name}")
                            else:
                                name = "Unregistered"
                    else:
                        # Not a good quality face for registration
                        name = "Unregistered"
                        should_add_to_excel = False
                        print("Face not suitable for registration - poor angle or quality")

            print("The returned name:", name)
            
            # Add occlusion indication if needed
            display_name = name
            if is_occluded:
                display_name += " (Masked)"
            
            # Update attendance in Excel
            current_time = datetime.now()
            current_date = str(current_time.date())
            current_time_str = str(current_time.strftime("%H:%M:%S"))
            
            # Calculate time difference
            time_difference = current_time - start_time
            minutes_late = time_difference.total_seconds() / 60
            
            # Modified attendance update code to include anonymous persons
            if name != "Unregistered" and should_add_to_excel:
                # Check if the date column exists, if not create it
                if current_date not in df.columns:
                    df[current_date] = "Absent"  # Initialize with "Absent" for all
                
                # Check if this person is already in the dataframe for today
                # Find the student's row
                if name in df["Name"].values:
                    student_row = df[df["Name"] == name].index[0]
            
                    # Update the attendance status if currently absent
                    if df.at[student_row, current_date] == "Absent":
                        if minutes_late > 20:
                            df.at[student_row, current_date] = f"Late ({current_time_str})"
                        else:
                            df.at[student_row, current_date] = f"Present ({current_time_str})"
                else:    
                    # Add a new row for the student or anonymous person
                    new_row = {"Name": name, current_date: f"Present ({str(current_time_str)})"}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # If this is a new anonymous person, add to our tracking list
                    if name.startswith("Anonymous_") and name not in anonymous_in_excel:
                        anonymous_in_excel.append(name)
                        
                df.to_excel(ATTENDANCE_FILE, index=False)
            
            # Draw rectangle and label on the frame with confidence
            confidence_color = (0, 255, 0)  # Green for high confidence
            
            # For recognized students
            if name in known_names:
                confidence_text = f" [Known]"
            elif name.startswith("Anonymous_"):
                confidence_text = f" [Anon]"
                confidence_color = (255, 165, 0)  # Orange for anonymous
            else:
                confidence_text = f" [??]"
                confidence_color = (0, 0, 255)  # Red for unrecognized
                
            # Add mask indicator
            if is_occluded:
                confidence_text += " [Masked]"
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), confidence_color, 2)
            
            # Display name with confidence info
            label = f"{display_name}{confidence_text}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 2)


        # Display the frame
        cv2.imshow("Face Recognition Attendance", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Load or create the attendance sheet
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        print(f"Loaded attendance sheet with {len(df)} entries.")
    else:
        df = pd.DataFrame(columns=["Name"])
        print("Created new attendance sheet.")
    
    # Load existing encodings
    known_encodings, known_names = load_encodings(ENCODINGS_FILE)
    anonymous_encodings, anonymous_names = load_encodings(ANONYMOUS_FILE)
    
    # Set start time for attendance
    start_time = datetime.now()
    
    # Run the attendance system
    attendance_system(start_time) 