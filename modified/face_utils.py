import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Multiply
from tensorflow.keras.models import Model
from insightface.app import FaceAnalysis

# Path configurations
STUDENT_IMAGES_FOLDER = r"../../Student_Images"
ENCODINGS_FILE = r"../../encodings.pkl"
ANONYMOUS_FILE = r"../../anonymous_faces.pkl"
ANONYMOUS_FOLDER = r"../../anonymous_faces"
ATTENDANCE_FILE = r"../../Attendance_Folder/Attendance_sheet.xlsx"

# Recognition threshold
THRESHOLD = 0.95

# Enhance image using histogram equalization (for low-light scenarios)
def enhance_image(image):
    """Apply advanced image enhancement to improve visibility in various lighting conditions"""
    # Check if the image is backlit (bright background, dark foreground)
    is_backlit = detect_backlighting(image)
    
    if is_backlit:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle backlighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Convert to LAB color space to preserve colors while enhancing contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction to boost dark areas
        gamma = 1.5
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
        enhanced_image = cv2.LUT(enhanced_image, lookUpTable)
    else:
        # Default enhancement for normal lighting
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        enhanced_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return enhanced_image

def detect_backlighting(image):
    """
    Detect if an image has backlighting (bright background, dark foreground)
    
    Args:
        image: BGR image
        
    Returns:
        bool: True if the image is likely backlit
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness in the center vs edges
    h, w = gray.shape
    center_region = gray[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    edge_region = gray.copy()
    edge_region[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)] = 0
    
    # Get average brightness
    center_brightness = np.mean(center_region[center_region > 0])
    edge_brightness = np.mean(edge_region[edge_region > 0])
    
    # Check if edges are significantly brighter than center (backlit)
    return edge_brightness > (center_brightness * 1.5)

def is_blurry(image, threshold=150.0):
    """
    Detect if an image is blurry using the Laplacian variance method
    
    Args:
        image: BGR image
        threshold: Threshold for Laplacian variance (lower = more blurry)
        
    Returns:
        bool: True if the image is blurry
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Return True if variance is below threshold (blurry)
    return lap_var < threshold

# Normalization function: L2-normalize an embedding vector
def normalize(embedding):
    """Normalize embedding vector to unit length"""
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm

# Squeeze-and-Excitation (SE) block as a channel attention module for embeddings
def build_attention_model(input_dim=512, reduction=16):
    """Build an attention model to enhance face embeddings"""
    input_tensor = Input(shape=(input_dim,))
    # Squeeze: reduce dimensions
    x = Dense(input_dim // reduction, activation='relu')(input_tensor)
    # Excitation: produce channel-wise weights
    x = Dense(input_dim, activation='sigmoid')(x)
    # Recalibrate: multiply original embedding by the weights
    output_tensor = Multiply()([input_tensor, x])
    model = Model(inputs=input_tensor, outputs=output_tensor, name="SE_Attention")
    return model

def get_face_analyzer():
    """Initialize and return an InsightFace analyzer"""
    face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return face_analyzer

def process_image(image_path, face_analyzer, attention_model):
    """Process an image and return face embedding"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, f"ERROR: Unable to read {image_path}"
    
    # Enhance and convert to RGB
    img = enhance_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = face_analyzer.get(img_rgb)
    
    if not faces:
        return None, f"No face detected in {image_path}"
    
    # Use first face
    face = faces[0]
    
    # Apply attention model to enhance embedding
    embedding = attention_model.predict(face.embedding[np.newaxis, :])[0]
    embedding = normalize(embedding)
    
    return embedding, None

def save_encodings(encodings_file, encodings, names):
    """Save encodings and names to a pickle file"""
    with open(encodings_file, 'wb') as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"Saved {len(encodings)} face encodings to {encodings_file}")

def load_encodings(encodings_file):
    """Load encodings and names from a pickle file"""
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            if os.path.getsize(encodings_file) > 0:
                encodings = pickle.load(f)
                return encodings["encodings"], encodings["names"]
            else:
                with open(encodings_file, "wb") as f:
                    save_encodings(encodings_file, [], [])
                return [], []
    return [], []

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two normalized embeddings"""
    return np.linalg.norm(a - b)

def add_padding_to_bbox(bbox, frame_shape, pad=10):
    """Add padding to bounding box while respecting image boundaries"""
    x, y, x2, y2 = bbox
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x2 + pad, frame_shape[1])
    y2 = min(y2 + pad, frame_shape[0])
    return x1, y1, x2, y2

def process_batch_embeddings(embeddings, attention_model, occlusion_flags=None):
    """
    Process multiple face embeddings in a single batch for efficiency
    
    Args:
        embeddings: List or numpy array of face embeddings
        attention_model: The attention model to apply to embeddings
        occlusion_flags: Optional list of boolean flags indicating if each face is occluded
    
    Returns:
        List of normalized embeddings after attention processing
    """
    if not embeddings:
        return []
        
    # Convert to numpy array if it's a list
    embeddings_array = np.array(embeddings)
    
    # If we have occlusion information, process occluded and non-occluded faces separately
    if occlusion_flags is not None and len(occlusion_flags) == len(embeddings):
        processed_embeddings = []
        
        # Get indices of occluded and non-occluded faces
        occluded_indices = [i for i, flag in enumerate(occlusion_flags) if flag]
        non_occluded_indices = [i for i, flag in enumerate(occlusion_flags) if not flag]
        
        # Process non-occluded faces with attention model
        if non_occluded_indices:
            non_occluded_embeddings = np.array([embeddings_array[i] for i in non_occluded_indices])
            non_occluded_processed = attention_model.predict(non_occluded_embeddings)
            
            # Place processed embeddings back in the right order
            for src_idx, original_idx in enumerate(non_occluded_indices):
                processed_embeddings.append(normalize(non_occluded_processed[src_idx]))
        
        # For occluded faces, skip attention model and just normalize directly
        for idx in occluded_indices:
            processed_embeddings.append(normalize(embeddings_array[idx]))
            
        return processed_embeddings
    else:
        # Process all embeddings with attention model
        processed_embeddings = attention_model.predict(embeddings_array)
        normalized_embeddings = [normalize(emb) for emb in processed_embeddings]
        return normalized_embeddings

def batch_process_faces(faces, attention_model):
    """
    Process a batch of detected faces
    
    Args:
        faces: List of face objects from face_analyzer.get()
        attention_model: The attention model to apply to embeddings
    
    Returns:
        List of normalized embeddings after attention processing
    """
    if not faces:
        return []
    
    # Validate faces before processing
    valid_faces = []
    occlusion_flags = []
    
    for face in faces:
        # Skip faces with invalid dimensions
        if face is None or not hasattr(face, 'embedding') or face.embedding is None:
            print("Invalid face object, skipping")
            continue
            
        # Skip faces with extremely small bounding boxes
        if hasattr(face, 'bbox'):
            bbox = face.bbox.astype(int)
            if bbox is None or len(bbox) != 4:
                print("Invalid bbox, skipping")
                continue
                
            x, y, x2, y2 = bbox
            if (x2 - x) < 30 or (y2 - y) < 30:
                print(f"Face bounding box too small: {x2-x}x{y2-y}, skipping")
                continue
        
        # Skip non-frontal faces based on pose angles
        if hasattr(face, 'pose'):
            # Extract roll, yaw, pitch from pose
            roll, yaw, pitch = face.pose
            if abs(yaw) > 30 or abs(pitch) > 20:
                print(f"Face angle too steep: yaw={yaw:.1f}°, pitch={pitch:.1f}°, skipping")
                continue
                
        # Check if face is occluded (will skip attention model if true)
        is_occluded, _, _ = detect_mask_or_occlusion(face)
        valid_faces.append(face)
        occlusion_flags.append(is_occluded)
    
    if not valid_faces:
        return []
        
    # Extract embeddings from valid faces
    face_embeddings = [face.embedding for face in valid_faces]
    
    # Process embeddings in batch, with occlusion information
    return process_batch_embeddings(face_embeddings, attention_model, occlusion_flags)

def is_good_quality_face(face):
    """
    Check if a face is of good quality for recognition
    
    Args:
        face: Face object from face_analyzer.get()
        
    Returns:
        bool: True if the face is suitable for recognition
    """
    # Check if face has valid dimensions
    if face is None or not hasattr(face, 'embedding') or face.embedding is None:
        return False
    
    # Check face angles (must be frontal)
    if hasattr(face, 'pose'):
        roll, yaw, pitch = face.pose
        if abs(yaw) > 30 or abs(pitch) > 20:
            return False
    
    # Check face size
    if hasattr(face, 'bbox'):
        bbox = face.bbox.astype(int)
        if bbox is None or len(bbox) != 4:
            return False
            
        x, y, x2, y2 = bbox
        if (x2 - x) < 30 or (y2 - y) < 30:
            return False
            
        # Blur check removed
                
    # Check for heavy occlusion - don't register faces with high occlusion
    is_occluded, occlusion_type, occlusion_score = detect_mask_or_occlusion(face)
    if is_occluded and occlusion_score > 0.4:  # More than 40% occluded
        return False
    
    # Face passed all checks
    return True

def detect_mask_or_occlusion(face):
    """
    Detect if a face is wearing a mask or is partially occluded
    
    Args:
        face: Face object from face_analyzer.get()
        
    Returns:
        tuple: (is_occluded, occlusion_type, occlusion_score)
    """
    # Default values
    is_occluded = False
    occlusion_type = "none"
    occlusion_score = 0.0
    
    # Check if face has keypoints
    if not hasattr(face, 'kps') or face.kps is None:
        return (False, "unknown", 0.0)
    
    # Get keypoints - these are nose, eyes, mouth corners
    keypoints = face.kps
    
    # Count visible keypoints
    total_keypoints = len(keypoints)
    if total_keypoints == 0:
        return (False, "unknown", 0.0)
    
    # Calculate percentage of face visible based on keypoints
    expected_keypoints = 5  # InsightFace usually detects 5 keypoints
    visible_keypoints = 0
    
    # Count valid keypoints (non-zero coordinates)
    for kp in keypoints:
        if kp[0] > 0 and kp[1] > 0:  # Valid x,y coordinates
            visible_keypoints += 1
    
    # Calculate basic visibility score
    visibility_ratio = visible_keypoints / expected_keypoints
    
    # In InsightFace, keypoints are typically ordered as:
    # [right_eye, left_eye, nose, right_mouth, left_mouth]
    
    # Check if lower face (mouth) keypoints are missing or have unusual y-coordinates
    # Get average y-coordinate of eyes
    if total_keypoints >= 2:
        eye_y_avg = (keypoints[0][1] + keypoints[1][1]) / 2
        
        # Check mouth keypoints if they exist
        if total_keypoints >= 5:
            mouth_y_avg = (keypoints[3][1] + keypoints[4][1]) / 2
            
            # If mouth keypoints are too close to eyes or missing, likely masked
            expected_mouth_eye_ratio = 0.20  # Reduced from 0.25 to detect more partial occlusions
            eye_to_mouth_distance = mouth_y_avg - eye_y_avg
            face_height = face.bbox[3] - face.bbox[1]
            
            if eye_to_mouth_distance < (face_height * expected_mouth_eye_ratio):
                is_occluded = True
                occlusion_type = "mask"
                occlusion_score = 0.8  # 80% occluded
                
                # Additional check for partial face covering
                if total_keypoints >= 3 and keypoints[2][0] > 0:  # If nose is visible
                    occlusion_score = 0.6  # 60% occluded - Less severe if nose is visible
                    occlusion_type = "partial_mask"
    
    # Also check asymmetry in keypoints which often indicates partial covering
    if total_keypoints >= 5:
        # Check horizontal balance of face (asymmetry can indicate partial covering)
        left_side_x = keypoints[1][0]  # Left eye x
        right_side_x = keypoints[0][0]  # Right eye x
        face_width = face.bbox[2] - face.bbox[0]
        
        # Calculate face center and compare with keypoint center
        face_center_x = face.bbox[0] + face_width/2
        keypoints_center_x = (left_side_x + right_side_x) / 2
        
        # If keypoints are significantly off-center, part of face may be covered
        center_offset = abs(face_center_x - keypoints_center_x) / face_width
        if center_offset > 0.15:  # If more than 15% off center
            is_occluded = True
            occlusion_type = "side_covered"
            occlusion_score = 0.5 + center_offset  # Range: 0.5-0.65 depending on offset
            
    # If we have face image, check pixel distribution for occlusion detection
    if hasattr(face, '_img') and face._img is not None:
        face_img = face._img
        if face_img.size > 0:
            # Calculate symmetry and distribution metrics as additional indicators
            try:
                # Split face image vertically into left and right halves
                height, width = face_img.shape[:2]
                left_half = face_img[:, :width//2]
                right_half = face_img[:, width//2:]
                
                # Calculate color histograms
                left_hist = cv2.calcHist([left_half], [0], None, [256], [0, 256])
                right_hist = cv2.calcHist([right_half], [0], None, [256], [0, 256])
                
                # Compare histograms for asymmetry
                hist_similarity = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
                
                # Low similarity indicates asymmetric occlusion (side covering)
                if hist_similarity < 0.5:
                    asymmetry_score = (1 - hist_similarity) * 0.5
                    if asymmetry_score > 0.2:  # Significant asymmetry
                        is_occluded = True
                        occlusion_type = "asymmetric_occlusion"
                        # Combine with previous score if it makes it higher
                        occlusion_score = max(occlusion_score, 0.5 + asymmetry_score)
            except:
                # Skip on any error processing the image
                pass
                
    # Enhance occlusion score using visibility ratio
    if visibility_ratio < 1.0:
        missing_keypoints_score = (1 - visibility_ratio) * 0.8
        occlusion_score = max(occlusion_score, missing_keypoints_score)
        
        # If more than 40% of keypoints are missing, consider occluded
        if visibility_ratio < 0.6:
            is_occluded = True
            if occlusion_type == "none":
                occlusion_type = "partial_occlusion"
    
    return (is_occluded, occlusion_type, occlusion_score)

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two normalized embeddings
    
    Args:
        a: First embedding
        b: Second embedding
        
    Returns:
        float: Cosine similarity (1.0 means identical, -1.0 means opposite)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def adaptive_threshold(face, base_threshold=THRESHOLD):
    """
    Calculate an adaptive threshold based on face quality
    
    Args:
        face: Face object from face_analyzer.get()
        base_threshold: The base threshold value
        
    Returns:
        float: Adjusted threshold
    """
    # Start with base threshold
    threshold = base_threshold
    
    # Check for occlusion
    is_occluded, occlusion_type, occlusion_score = detect_mask_or_occlusion(face)
    
    # If face is occluded, make threshold more lenient
    if is_occluded:
        # Adjust threshold based on occlusion score (0.0-1.0)
        # The higher the occlusion, the more lenient we are
        threshold_adjustment = 0.35 * occlusion_score  # Up to 35% more lenient (increased from 25%)
        threshold = base_threshold * (1 - threshold_adjustment)
        print(f"Adjusting threshold for {occlusion_type}: {threshold:.3f} (was {base_threshold:.3f})")
    
    # Check face pose
    if hasattr(face, 'pose'):
        roll, yaw, pitch = face.pose
        
        # If face is turned, make threshold more lenient
        pose_adjustment = (abs(yaw) / 30) * 0.15  # Up to 15% more lenient (increased from 10%)
        threshold = threshold * (1 - pose_adjustment)
    
    return threshold

# Face tracking for temporal consistency
class FaceTracker:
    """
    Track faces across multiple frames for more stable recognition
    """
    def __init__(self, max_history=15, similarity_threshold=0.75):
        self.face_history = {}  # id -> list of embeddings
        self.position_history = {}  # id -> list of positions (bbox centers)
        self.last_seen = {}     # id -> frame number
        self.current_frame = 0
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.next_id = 0
        self.occlusion_history = {}  # id -> list of occlusion states
        self.recent_matches = {}     # id -> list of matched names
        self.velocity = {}  # id -> (dx, dy) estimated motion vector
    
    def get_track_id(self, embedding, frame_number, face=None):
        """
        Get track ID for a face embedding
        
        Args:
            embedding: Face embedding
            frame_number: Current frame number
            face: Face object to check for occlusion
            
        Returns:
            int: Track ID
        """
        self.current_frame = frame_number
        
        # Check for occlusion
        is_occluded = False
        if face is not None:
            is_occluded, _, _ = detect_mask_or_occlusion(face)
        
        # Get face position (center point of bbox)
        current_position = None
        if face is not None and hasattr(face, 'bbox'):
            bbox = face.bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            current_position = (center_x, center_y)
        
        # Find closest track by both appearance (embedding) and motion consistency
        best_match = None
        best_score = -1  # Combined score
        
        # Use a lower similarity threshold for occluded faces
        current_similarity_threshold = self.similarity_threshold
        if is_occluded:
            current_similarity_threshold *= 0.85  # 15% more lenient for occluded faces
        
        for track_id in self.face_history:
            # Skip if no embeddings
            if not self.face_history[track_id]:
                continue
            
            # Calculate appearance similarity
            avg_embedding = np.mean(self.face_history[track_id], axis=0)
            appearance_similarity = cosine_similarity(embedding, avg_embedding)
            
            # Calculate position similarity if we have position data
            position_similarity = 0
            if current_position and track_id in self.position_history and self.position_history[track_id]:
                # Get last position
                last_positions = self.position_history[track_id]
                last_position = last_positions[-1]
                
                # If we have velocity data, predict current position based on last position and velocity
                if track_id in self.velocity:
                    dx, dy = self.velocity[track_id]
                    frames_elapsed = frame_number - self.last_seen[track_id]
                    # Don't predict too far in the future
                    if frames_elapsed <= 5:  
                        predicted_x = last_position[0] + dx * frames_elapsed
                        predicted_y = last_position[1] + dy * frames_elapsed
                        predicted_position = (predicted_x, predicted_y)
                    else:
                        predicted_position = last_position
                else:
                    predicted_position = last_position
                
                # Calculate Euclidean distance between predicted and actual positions
                distance = np.sqrt((predicted_position[0] - current_position[0])**2 + 
                                  (predicted_position[1] - current_position[1])**2)
                
                # Convert distance to similarity (closer = higher similarity)
                # Normalize based on face size (assuming face width is about 100 pixels)
                face_size = 100
                if hasattr(face, 'bbox'):
                    face_size = face.bbox[2] - face.bbox[0]  # face width
                
                # Similarity decreases with distance, scaled by face size
                # More lenient for quick motions (larger face_size multiplier)
                max_distance = face_size * 1.5  # Allow more movement (up to 1.5x face width)
                position_similarity = max(0, 1 - (distance / max_distance))
            
            # For heavily occluded faces, check if this track was recently occluded too
            occlusion_bonus = 0
            if is_occluded and track_id in self.occlusion_history:
                recent_occlusions = self.occlusion_history[track_id]
                if recent_occlusions and any(recent_occlusions[-3:]):  # Check the last 3 frames
                    # Boost similarity for recently occluded tracks to maintain identity
                    occlusion_bonus = 0.05  # Slight boost to favor previously occluded tracks
            
            # Calculate final combined score
            # For very fast motions, prioritize appearance over position
            appearance_weight = 0.7
            position_weight = 0.3
            
            # If we have no position data, rely entirely on appearance
            if position_similarity == 0:
                combined_score = appearance_similarity + occlusion_bonus
            else:
                combined_score = (appearance_weight * appearance_similarity) + \
                                (position_weight * position_similarity) + \
                                occlusion_bonus
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = track_id
        
        appearance_threshold = current_similarity_threshold
        
        # If good match found, update history
        if best_match is not None and best_score > appearance_threshold:
            track_id = best_match
            
            # Update embedding history
            history = self.face_history[track_id]
            history.append(embedding)
            
            # Keep only recent history
            if len(history) > self.max_history:
                history.pop(0)
            
            # Update position history
            if current_position:
                if track_id not in self.position_history:
                    self.position_history[track_id] = []
                
                self.position_history[track_id].append(current_position)
                
                # Keep only recent positions
                if len(self.position_history[track_id]) > self.max_history:
                    self.position_history[track_id].pop(0)
                
                # Update velocity if we have at least 2 positions
                if len(self.position_history[track_id]) >= 2:
                    pos1 = self.position_history[track_id][-2]
                    pos2 = self.position_history[track_id][-1]
                    frames_between = 1  # Assume consecutive frames
                    
                    # Calculate velocity as pixels per frame
                    dx = (pos2[0] - pos1[0]) / frames_between
                    dy = (pos2[1] - pos1[1]) / frames_between
                    
                    # Apply some damping to avoid overreacting to sudden changes
                    damping = 0.7
                    if track_id in self.velocity:
                        old_dx, old_dy = self.velocity[track_id]
                        dx = old_dx * (1-damping) + dx * damping
                        dy = old_dy * (1-damping) + dy * damping
                    
                    # Store velocity
                    self.velocity[track_id] = (dx, dy)
            
            # Update occlusion history
            if track_id not in self.occlusion_history:
                self.occlusion_history[track_id] = []
            self.occlusion_history[track_id].append(is_occluded)
            if len(self.occlusion_history[track_id]) > self.max_history:
                self.occlusion_history[track_id].pop(0)
            
            # Update last seen
            self.last_seen[track_id] = frame_number
        else:
            # Create new track
            track_id = self.next_id
            self.next_id += 1
            self.face_history[track_id] = [embedding]
            if current_position:
                self.position_history[track_id] = [current_position]
            self.occlusion_history[track_id] = [is_occluded]
            self.last_seen[track_id] = frame_number
        
        # Clean up old tracks
        self._clean_old_tracks()
        
        return track_id
    
    def update_track_name(self, track_id, name):
        """
        Update the name associated with a track for consistency
        
        Args:
            track_id: Track ID
            name: Name to associate with the track
        """
        if track_id not in self.recent_matches:
            self.recent_matches[track_id] = []
            
        self.recent_matches[track_id].append(name)
        
        # Keep only recent history
        if len(self.recent_matches[track_id]) > self.max_history:
            self.recent_matches[track_id].pop(0)
    
    def get_consistent_name(self, track_id, current_name):
        """
        Get the most consistent name for a track to avoid jumping between identities
        
        Args:
            track_id: Track ID
            current_name: Currently detected name
            
        Returns:
            str: Most consistent name
        """
        if track_id not in self.recent_matches:
            return current_name
            
        # Count occurrences of each name in recent history
        name_counts = {}
        for name in self.recent_matches[track_id]:
            if name not in name_counts:
                name_counts[name] = 0
            name_counts[name] += 1
            
        # If current name is already in history, give it a slight preference
        if current_name in name_counts:
            name_counts[current_name] += 0.5
            
        # Determine the most frequent name
        most_frequent_name = current_name
        max_count = 0
        
        for name, count in name_counts.items():
            if count > max_count:
                max_count = count
                most_frequent_name = name
                
        return most_frequent_name
    
    def _clean_old_tracks(self, max_age=30):
        """
        Remove tracks that haven't been seen recently
        
        Args:
            max_age: Maximum number of frames a track can be unseen
        """
        tracks_to_remove = []
        
        for track_id, last_frame in self.last_seen.items():
            if self.current_frame - last_frame > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.face_history[track_id]
            del self.last_seen[track_id]
            if track_id in self.occlusion_history:
                del self.occlusion_history[track_id]
            if track_id in self.recent_matches:
                del self.recent_matches[track_id]
            if track_id in self.position_history:
                del self.position_history[track_id]
            if track_id in self.velocity:
                del self.velocity[track_id]

def find_best_match(embedding, known_embeddings, known_names, face=None, use_cosine=True, base_threshold=THRESHOLD):
    """
    Find the best match for a face embedding among known embeddings
    
    Args:
        embedding: Face embedding to match
        known_embeddings: List of known embeddings
        known_names: List of known names corresponding to embeddings
        face: Optional face object for adaptive thresholding
        use_cosine: Whether to use cosine similarity (True) or Euclidean distance (False)
        base_threshold: Base threshold for matching
        
    Returns:
        tuple: (match_name, match_distance, is_match)
    """
    if not known_embeddings or not known_names:
        return ("Unknown", float("inf"), False)
    
    # Determine threshold
    threshold = adaptive_threshold(face, base_threshold) if face is not None else base_threshold
    
    # Choose similarity function
    similarity_func = cosine_similarity if use_cosine else lambda a, b: -euclidean_distance(a, b)
    
    # Find best match
    best_match = None
    best_similarity = -float("inf") if use_cosine else float("inf")
    
    for known_embedding, known_name in zip(known_embeddings, known_names):
        # Calculate similarity
        similarity = similarity_func(embedding, known_embedding)
        
        # For cosine similarity, higher is better
        # For Euclidean distance, lower is better (negative values in similarity_func)
        if (use_cosine and similarity > best_similarity) or (not use_cosine and similarity < best_similarity):
            best_similarity = similarity
            best_match = known_name
    
    # Convert to normalized score
    if use_cosine:
        # Cosine similarity: 1.0 is best, already normalized
        score = best_similarity
        is_match = score > threshold
    else:
        # Euclidean: negate to get positive value, normalize against threshold
        score = -best_similarity  # Now lower is better again
        is_match = score < threshold
    
    return (best_match, score, is_match) 