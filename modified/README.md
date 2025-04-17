# Using InsightFace for Complete Face Recognition Pipeline

This document explains how we've updated the Attendance System to use InsightFace for both face detection and face embeddings, completely replacing both MTCNN and FaceNet.

## Recent Updates

We've made the following improvements:

1. **Complete InsightFace Integration**:

   - Now using InsightFace for both face detection AND face embeddings
   - Removed dependency on FaceNet for embeddings
   - Removed dependency on face_recognition library

2. **Bounding Box Enhancement**:

   - Added padding to facial bounding boxes for improved recognition
   - Ensures faces are properly captured with surrounding context

3. **Simplified Architecture**:
   - Single embedding format throughout the pipeline
   - More consistent approach using one face recognition library
   - Still using our custom attention model (SE-block) to enhance embeddings

## Setup Instructions

### 1. Create Student Embeddings

Before running the attendance system, you need to create embeddings for the students you want to recognize:

1. Place student images in the `Student_Images` folder:

   - Use one clear face image per student
   - Name the file with the student's name (e.g., `john_smith.jpg`)
   - The filename (without extension) will be used as the student's name

2. Run the embedding creator script:

   ```
   python create_student_embeddings.py
   ```

3. The script will:
   - Process each image in the folder
   - Detect faces using InsightFace
   - Generate enhanced embeddings through our attention model
   - Save the embeddings to `encodings.pkl`

### 2. Run the Attendance System

After creating the embeddings, you can run the attendance system:

```
python modified_attendance_system.py
```

## Implementation Details

### InsightFace Setup

```python
face_analyzer = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# Specifically request the recognition model
face_analyzer.prepare(ctx_id=0, det_size=(640, 640), rec_model='arcface_r100_v1')
```

### Embedding Extraction

```python
# Direct use of InsightFace embeddings with our attention model
embedding = attention_model.predict(face.embedding[np.newaxis, :])[0]
embedding = normalize(embedding)
```

### Bounding Box Padding

```python
# Add padding to the bounding box
pad = 10
x1 = max(x - pad, 0)
y1 = max(y - pad, 0)
x2 = min(x2 + pad, rgb_frame.shape[1])
y2 = min(y2 + pad, rgb_frame.shape[0])
face_img = rgb_frame[y1:y2, x1:x2]
```

## Benefits

1. **Improved Efficiency**:

   - Single library for both detection and recognition
   - Optimized pipeline with fewer preprocessing steps
   - Faster processing, especially on GPU

2. **Enhanced Accuracy**:

   - InsightFace provides state-of-the-art facial embeddings (ArcFace)
   - Padding around faces improves recognition quality
   - Our attention model further enhances the embeddings

3. **Simplified Architecture**:
   - Cleaner code with fewer dependencies
   - More maintainable with consistent approaches
   - Easier to update or enhance in the future

## Usage Example

### Creating New Student Records

1. Add a student photo to the `Student_Images` folder (name it with the student's name)
2. Run `python create_student_embeddings.py` to update the embedding database
3. The system will now recognize this student during attendance tracking

### Tracking Attendance

Run the system with:

```
python modified_attendance_system.py
```

The system will:

- Detect faces in the webcam feed
- Match against known student embeddings
- Record attendance in the Excel sheet
- Track unknown persons as anonymous entries
