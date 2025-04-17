# Attendance System with Machine Vision

A comprehensive attendance management system using advanced facial recognition technology. This system combines machine learning-based face recognition with a modern web interface for managing student attendance records.

## Features

- **Advanced Facial Recognition**: Uses InsightFace and custom attention models for robust face detection and recognition
- **Low-Light Enhancement**: Specialized image processing for improved recognition in various lighting conditions
- **Web Interface**: Modern React frontend with Node.js backend
- **Excel Integration**: Automatic attendance recording and synchronization with Excel spreadsheets
- **Tracking System**: Records both known students and anonymous faces for later identification
- **Quality Assessment**: Detects blur, occlusion, and poor lighting for optimal face recognition

## System Architecture

The project consists of three main components:

1. **Core Recognition Engine** - Python-based facial recognition system
2. **Backend Server** - Node.js API for handling requests and managing data
3. **Frontend Application** - React-based user interface

## Prerequisites

- Python 3.8+ (with pip)
- Node.js 14+ (with npm)
- Webcam for facial recognition
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/attendance-system-machine-vision.git
cd attendance-system-machine-vision
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Backend

```bash
cd attendance-backend
npm install
```

### 4. Set Up Frontend

```bash
cd attendance-frontend/my-react-app
npm install
```

## Usage

You can use this system in multiple ways:

### Option 1: Core Recognition Engine (Python)

1. **Create Student Embeddings**:

   - Place student images in the `Student_Images` folder (one clear face image per student)
   - Name each file with the student's name (e.g., `john_smith.jpg`)
   - Run the embedding creator:
     ```bash
     python modified/create_student_embeddings.py
     ```

2. **Run the Attendance System**:

   ```bash
   python modified/modified_attendance_system.py
   ```

   This will:

   - Open your webcam
   - Detect and recognize faces
   - Record attendance in the Excel sheet
   - Track unknown persons as anonymous entries

### Option 2: Web Application

1. **Start the Backend Server**:

   ```bash
   cd attendance-backend
   npm run dev
   ```

   The server will run on http://localhost:5000

2. **Start the Frontend**:

   ```bash
   cd attendance-frontend/my-react-app
   npm run dev
   ```

   The application will be available at http://localhost:5173

3. **Access the Web Interface** in your browser and:
   - View attendance records by date
   - Add new students with photo upload
   - Update student attendance status and time
   - View recognition statistics

## Key Components Explained

### Face Recognition Pipeline

1. **Face Detection**: Using InsightFace for robust face detection
2. **Embedding Generation**: Converting faces to 512-dimensional embeddings
3. **Attention Enhancement**: Applying squeeze-and-excitation blocks to improve embeddings
4. **Matching**: Comparing against known faces with adaptive thresholds

### Face Tracking

The system implements a temporal tracking mechanism that:

- Maintains identity consistency across video frames
- Handles temporary occlusions
- Improves recognition accuracy through voting

### Web API Endpoints

- `GET /api/attendance`: Get all attendance data
- `PUT /api/attendance`: Update a student's attendance for a specific date
- `POST /api/students`: Add a new student with photo
- `GET /api/photos/:filename`: Retrieve a student's photo

## Project Structure

```
Attendance_System/
├── Attendance_Folder/        # Excel attendance sheets
├── Student_Images/           # Student profile photos
├── anonymous_faces/          # Stored unknown faces
├── attendance-backend/       # Node.js backend server
├── attendance-frontend/      # React frontend
├── modified/                 # Enhanced recognition engine
│   ├── face_utils.py         # Face recognition utilities
│   └── modified_attendance_system.py  # Main recognition application
├── Prototype.ipynb           # Development notebook
└── requirements.txt          # Python dependencies
```

## Excel Format

The Excel attendance sheet has the following format:

- First column: Student names
- First row: Date headers
- Cell values: Attendance status - "Present", "Absent", "Late" with optional time in parentheses

## Technologies Used

- **Computer Vision**: OpenCV, InsightFace
- **Deep Learning**: TensorFlow, Keras
- **Web Backend**: Node.js, Express
- **Web Frontend**: React, Axios
- **Data Storage**: Excel, Pickle

## License

[MIT License](LICENSE)

## Contributors

- [Hari Kishan](https://github.com/Harii2K4)

## Acknowledgments

- InsightFace for state-of-the-art face recognition
- TensorFlow team for deep learning framework
- OpenCV contributors
