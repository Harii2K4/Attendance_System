# Attendance System Backend

This backend server integrates with the React frontend to provide attendance management with facial recognition features.

## Prerequisites

Before running the backend server, make sure you have the following installed:

1. Node.js 14+ and npm
2. Python 3.6+ with pip
3. Required Python libraries (face_recognition, numpy)

## Setup

1. **Install Node.js dependencies:**

   ```
   npm install
   ```

2. **Install required Python libraries:**

   ```
   npm run install-face-recognition
   ```

   This will install the `face_recognition` and `numpy` Python packages required for facial recognition.

   Alternatively, you can install them directly using pip:

   ```
   pip install face_recognition numpy
   ```

## How Face Recognition Works

The backend server implements facial recognition to compare new student photos with existing anonymous records:

1. When a new student is added with their photo, the system:

   - Extracts facial encodings from the uploaded photo
   - Compares these encodings with stored anonymous encodings
   - If a match is found, the anonymous record is replaced with the new student's info
   - If no match is found, a new student record is created

2. The system manages two encoding files:
   - `encodings.pickle`: Contains known student face encodings
   - `anonymous_encodings.pickle`: Contains unidentified face encodings

## API Endpoints

- `GET /api/attendance`: Retrieves all attendance data
- `PUT /api/attendance`: Updates attendance status for a student
- `POST /api/students`: Adds a new student with face recognition
- `GET /api/photos/:filename`: Retrieves student photos

## File Structure

- `server.js`: Main server file with API endpoints and facial recognition logic
- `load_pickle.py`, `save_pickle.py`, etc.: Temporary Python scripts created at runtime

## Running the Server

```
npm run dev
```

The server will run on port 5000 by default.
