const express = require('express');
const cors = require('cors');
const multer = require('multer');
const exceljs = require('exceljs');
const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

// Create Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Create a temp directory for storing temporary files
const TEMP_DIR = path.join(__dirname, 'temp');
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
}

// Create a workers directory for the standalone face processing scripts
const WORKERS_DIR = path.join(__dirname, 'workers');
if (!fs.existsSync(WORKERS_DIR)) {
  fs.mkdirSync(WORKERS_DIR, { recursive: true });
}

// Create a standalone face encoding worker script that won't be monitored by nodemon
const faceWorkerPath = path.join(WORKERS_DIR, 'face_worker.py');
fs.writeFileSync(faceWorkerPath, `
import sys
import os
import json
import numpy as np
import cv2
import tensorflow as tf
import pickle
from datetime import datetime

# Add the parent directory to path so we can import face_utils
sys.path.append('${path.join(__dirname, '..').replace(/\\/g, '\\\\')}')
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
`);

console.log('Created standalone face worker script');

// Helper function to run the face worker script
function runFaceWorker(command, args) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(TEMP_DIR, `worker_output_${Date.now()}.json`);
    const allArgs = [faceWorkerPath, command, ...args, outputPath];
    
    console.log(`Running face worker: python ${allArgs.join(' ')}`);
    
    const workerProcess = spawn('python', allArgs);
    
    workerProcess.stdout.on('data', (data) => {
      console.log(`Worker stdout: ${data.toString()}`);
    });
    
    workerProcess.stderr.on('data', (data) => {
      console.error(`Worker stderr: ${data.toString()}`);
    });
    
    workerProcess.on('close', (code) => {
      console.log(`Worker process exited with code ${code}`);
      if (code === 0) {
        // Read the output file
        try {
          const result = JSON.parse(fs.readFileSync(outputPath, 'utf-8'));
          fs.unlinkSync(outputPath); // Clean up the output file
          resolve(result);
        } catch (error) {
          console.error('Error reading worker output:', error);
          reject(new Error(`Failed to read worker output: ${error.message}`));
        }
      } else {
        reject(new Error(`Worker process failed with code ${code}`));
      }
    });
  });
}

// Helper function to get face encodings using the worker script
async function getFaceEncodings(imagePath) {
  try {
    console.log(`Getting face encodings for: ${imagePath}`);
    const result = await runFaceWorker('encode', [imagePath]);
    
    if (result && result.success && result.encodings && result.encodings.length > 0) {
      console.log(`Successfully encoded ${result.encodings.length} faces`);
      return result.encodings;
    } else {
      console.error('Face encoding failed:', result);
      return [];
    }
  } catch (error) {
    console.error('Error in face encoding:', error);
    return [];
  }
}

// Create a .nodemonignore file in the project root to prevent unwanted restarts
// This is more reliable than nodemon.json
const nodemonIgnorePath = path.join(__dirname, '.nodemonignore');
fs.writeFileSync(nodemonIgnorePath, `
# Ignore temp directory
temp/*
# Ignore all Python files
*.py
# Ignore JSON files
*.json
# Ignore pickle files
*.pkl
# Ignore Student Images
../Student_Images/*
# Ignore Excel files
*.xlsx
`);
console.log('Created .nodemonignore file to prevent unwanted restarts');

// Try to disable file watching for this process
if (process.env.NODEMON) {
  console.log('Running under nodemon - attempting to stabilize file watching');
  try {
    // Send a signal to nodemon to reduce watch sensitivity
    process.send({ type: 'process:msg', data: { action: 'config:update', config: { delay: 5000 } } });
  } catch (err) {
    console.log('Could not communicate with nodemon parent process');
  }
}

// Middleware
app.use(cors());
app.use(express.json());

// Path to attendance Excel file
const ATTENDANCE_FILE = path.join(__dirname, '..', 'Attendance_Folder', 'Attendance_sheet.xlsx');
const STUDENT_IMAGES_FOLDER = path.join(__dirname, '..', 'Student_Images');
const KNOWN_ENCODINGS_FILE = path.join(__dirname, '..', 'encodings.pkl');
const ANONYMOUS_FILE = path.join(__dirname, '..', 'anonymous_faces.pkl');

// Create a path to face_utils.py
const FACE_UTILS_PY = path.join(__dirname, '..', 'modified', 'face_utils.py');

// Ensure the Student_Images folder exists
if (!fs.existsSync(STUDENT_IMAGES_FOLDER)) {
  fs.mkdirSync(STUDENT_IMAGES_FOLDER, { recursive: true });
}

// Configure multer for file upload with more flexible options
const upload = multer({
  storage: multer.diskStorage({
    destination: function (req, file, cb) {
      cb(null, STUDENT_IMAGES_FOLDER);
    },
    filename: function (req, file, cb) {
      // Use student name as filename
      const studentName = req.body.name || 'unknown';
      const fileExt = path.extname(file.originalname);
      
      // Convert spaces to underscores and remove special characters
      const sanitizedName = studentName.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
      
      cb(null, `${sanitizedName}${fileExt}`);
    }
  }),
  fileFilter: function (req, file, cb) {
    // Accept only image files
    if (!file.originalname.match(/\.(jpg|jpeg|png|gif)$/i)) {
      return cb(new Error('Only image files are allowed!'), false);
    }
    cb(null, true);
  }
});

// Helper function to load encodings from a pickle file
function loadEncodings(filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      return { encodings: [], names: [] };
    }
    
    // We need to use Python to load the pickle file since it contains numpy arrays
    // Use a temporary solution where we execute a Python script to convert the pickle to JSON
    const tempJsonPath = path.join(TEMP_DIR, `temp_encodings_${Date.now()}.json`);
    
    // Create a temporary Python script that imports face_utils
    const tempPyScript = path.join(TEMP_DIR, `load_pickle_${Date.now()}.py`);
    fs.writeFileSync(tempPyScript, `
import pickle
import json
import numpy as np
import os
import sys

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_pickle(pickle_path, json_path):
    if not os.path.exists(pickle_path):
        with open(json_path, 'w') as f:
            json.dump({"encodings": [], "names": []}, f)
        return
        
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)

if __name__ == "__main__":
    pickle_path = sys.argv[1]
    json_path = sys.argv[2]
    load_pickle(pickle_path, json_path)
    `);
    
    // Execute the Python script
    execSync(`python ${tempPyScript} "${filePath}" "${tempJsonPath}"`);
    
    // Read the resulting JSON
    const data = JSON.parse(fs.readFileSync(tempJsonPath, 'utf-8'));
    
    // Clean up temporary files
    fs.unlinkSync(tempPyScript);
    fs.unlinkSync(tempJsonPath);
    
    return data;
  } catch (error) {
    console.error('Error loading encodings:', error);
    return { encodings: [], names: [] };
  }
}

// Helper function to save encodings data to a pickle file
function saveEncodings(encodingsData) {
  try {
    // Create a temporary JSON file with the encoding data
    const timestamp = Date.now();
    const tempJsonPath = path.join(TEMP_DIR, `temp_encodings_${timestamp}.json`);
    const tempPyScript = path.join(TEMP_DIR, `save_encodings_${timestamp}.py`);
    
    // Write the encodings data to JSON
    fs.writeFileSync(tempJsonPath, JSON.stringify(encodingsData));
    
    // Create a Python script to convert the JSON to pickle
    fs.writeFileSync(tempPyScript, `
import json
import pickle
import numpy as np
import os
import sys

def convert_json_to_pickle(json_path, pickle_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    for i, student in enumerate(data.get('students', [])):
        if 'encodings' in student:
            for j, encoding in enumerate(student['encodings']):
                if encoding:  # Only convert if not None
                    data['students'][i]['encodings'][j] = np.array(encoding)
    
    # Save as pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    json_path = sys.argv[1]
    pickle_path = sys.argv[2]
    convert_json_to_pickle(json_path, pickle_path)
    `);
    
    // Execute the Python script
    execSync(`python ${tempPyScript} "${tempJsonPath}" "${KNOWN_ENCODINGS_FILE}"`);
    
    // Clean up temporary files
    fs.unlinkSync(tempJsonPath);
    fs.unlinkSync(tempPyScript);
    
    console.log('Encodings saved successfully');
  } catch (error) {
    console.error('Error saving encodings:', error);
  }
}

// Helper function to compare face encodings using face_utils.py functionality
function compareFaces(knownEncodings, faceEncoding) {
  try {
    // Create a temporary Python script for face comparison using face_utils
    const tempPyScript = path.join(TEMP_DIR, `compare_faces_${Date.now()}.py`);
    const tempKnownJsonPath = path.join(TEMP_DIR, `temp_known_encodings_${Date.now()}.json`);
    const tempFaceJsonPath = path.join(TEMP_DIR, `temp_face_encoding_${Date.now()}.json`);
    const tempResultsPath = path.join(TEMP_DIR, `temp_comparison_results_${Date.now()}.json`);
    
    // Write the encodings to temporary JSON files
    fs.writeFileSync(tempKnownJsonPath, JSON.stringify(knownEncodings));
    fs.writeFileSync(tempFaceJsonPath, JSON.stringify(faceEncoding));
    
    // Create script that uses face_utils.py functions
    fs.writeFileSync(tempPyScript, `
import sys
import os
import json
import numpy as np

# Add the correct path to the modified directory
sys.path.append('${path.join(__dirname, '..').replace(/\\/g, '\\\\')}')
from modified.face_utils import cosine_similarity

def compare_faces(known_encodings_path, face_encoding_path, results_path):
    with open(known_encodings_path, 'r') as f:
        known_encodings = json.load(f)
    
    with open(face_encoding_path, 'r') as f:
        face_encoding = json.load(f)
    
    # Convert lists back to numpy arrays
    known_encodings = [np.array(enc) for enc in known_encodings]
    face_encoding = np.array(face_encoding)
    
    # Calculate similarities using cosine_similarity from face_utils
    similarities = [cosine_similarity(face_encoding, enc) for enc in known_encodings]
    
    # Create matches based on threshold
    threshold = 0.75  # Adjust this to match face_utils.py THRESHOLD
    matches = [sim > threshold for sim in similarities]
    
    # Calculate face distances
    distances = [1.0 - sim for sim in similarities]
    
    # Combine results with distances for better analysis
    combined_results = {
        "matches": matches,
        "distances": distances,
        "similarities": similarities,
        "best_match_index": np.argmax(similarities) if similarities else -1,
        "best_match_similarity": max(similarities) if similarities else 0.0
    }
    
    with open(results_path, 'w') as f:
        json.dump(combined_results, f)

if __name__ == "__main__":
    known_encodings_path = sys.argv[1]
    face_encoding_path = sys.argv[2]
    results_path = sys.argv[3]
    compare_faces(known_encodings_path, face_encoding_path, results_path)
    `);
    
    // Execute the Python script
    execSync(`python ${tempPyScript} "${tempKnownJsonPath}" "${tempFaceJsonPath}" "${tempResultsPath}"`);
    
    // Read the comparison results
    const results = JSON.parse(fs.readFileSync(tempResultsPath, 'utf-8'));
    
    // Clean up temporary files
    fs.unlinkSync(tempPyScript);
    fs.unlinkSync(tempKnownJsonPath);
    fs.unlinkSync(tempFaceJsonPath);
    fs.unlinkSync(tempResultsPath);
    
    // If we have the new combined format, return just the matches
    if (results.matches && Array.isArray(results.matches)) {
      console.log("Face similarities:", results.similarities);
      return results.matches;
    }
    
    return results;
  } catch (error) {
    console.error('Error comparing faces:', error);
    return [];
  }
}

// Helper function to read attendance data from Excel
async function readAttendanceData() {
  const workbook = new exceljs.Workbook();
  await workbook.xlsx.readFile(ATTENDANCE_FILE);
  
  const worksheet = workbook.getWorksheet(1);
  
  // Parse the data
  const dates = [];
  const students = [];
  const attendance = {};
  
  // Get column names (dates)
  worksheet.getRow(1).eachCell((cell, colNumber) => {
    if (colNumber > 1) { // Skip the first column which contains "Name"
      const date = cell.value;
      if (date) {
        dates.push(date);
        attendance[date] = {};
      }
    }
  });
  
  // Get student data and attendance
  let rowNumber = 2; // Start from the second row (skip header)
  while (rowNumber <= worksheet.rowCount) {
    const row = worksheet.getRow(rowNumber);
    const name = row.getCell(1).value;
    
    if (name) {
      const studentId = rowNumber - 1; // Use the row index as ID
      
      // Add student to the list
      students.push({
        id: studentId,
        name: name,
        photo: null // We don't have photo information in the Excel
      });
      
      // Get attendance for each date
      dates.forEach((date, index) => {
        const cellValue = row.getCell(index + 2).value || 'Absent';
        
        // Parse the attendance status and time
        let status = 'Absent';
        let time = '';
        
        if (typeof cellValue === 'string') {
          const match = cellValue.match(/(\w+)\s*\(([^)]+)\)/);
          if (match) {
            status = match[1];
            time = match[2];
          } else {
            status = cellValue;
          }
        }
        
        attendance[date][studentId] = { status, time };
      });
    }
    
    rowNumber++;
  }
  
  return { dates, students, attendance };
}

// Helper function to write attendance data to Excel
async function updateAttendanceData(date, studentId, status, time) {
  const workbook = new exceljs.Workbook();
  await workbook.xlsx.readFile(ATTENDANCE_FILE);
  
  const worksheet = workbook.getWorksheet(1);
  
  // Find the column for the date
  let dateColumn = null;
  worksheet.getRow(1).eachCell((cell, colNumber) => {
    if (cell.value === date) {
      dateColumn = colNumber;
    }
  });
  
  if (!dateColumn) {
    throw new Error(`Date ${date} not found in the attendance sheet`);
  }
  
  // Update the attendance
  const rowNumber = parseInt(studentId) + 1; // Convert studentId to row number
  
  // Format the value with status and time if provided
  let cellValue = status;
  if (time && status !== 'Absent') {
    cellValue = `${status} (${time})`;
  }
  
  worksheet.getCell(rowNumber, dateColumn).value = cellValue;
  
  // Save the workbook
  await workbook.xlsx.writeFile(ATTENDANCE_FILE);
  
  return { success: true };
}

// Helper function to check if a file is accessible
function isFileAccessible(filePath, timeout = 10000) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    
    const checkAccess = () => {
      try {
        // Try to open the file with read-write access
        const fd = fs.openSync(filePath, 'r+');
        // If successful, close it immediately
        fs.closeSync(fd);
        resolve(true);
      } catch (error) {
        // If we've waited too long, give up
        if (Date.now() - startTime > timeout) {
          reject(new Error(`File ${filePath} is locked and couldn't be accessed after ${timeout}ms`));
          return;
        }
        
        // Otherwise wait a bit and try again
        setTimeout(checkAccess, 500);
      }
    };
    
    checkAccess();
  });
}

// Helper function to add a new student to Excel
async function addNewStudentToExcel(name) {
  // First check if the file is accessible
  try {
    await isFileAccessible(ATTENDANCE_FILE);
  } catch (error) {
    console.error('Excel file access error:', error);
    throw new Error('Excel file is locked. Please close any applications using it and try again.');
  }
  
  const workbook = new exceljs.Workbook();
  await workbook.xlsx.readFile(ATTENDANCE_FILE);
  
  const worksheet = workbook.getWorksheet(1);
  
  // Create a new row for the student
  const newRow = worksheet.addRow();
  newRow.getCell(1).value = name;
  
  // Set 'Absent' for all date columns
  let colNumber = 2;
  while (colNumber <= worksheet.columnCount) {
    newRow.getCell(colNumber).value = 'Absent';
    colNumber++;
  }
  
  // Save the workbook
  await workbook.xlsx.writeFile(ATTENDANCE_FILE);
  
  // Return the new student ID (row number - 1)
  return { id: worksheet.rowCount - 1 };
}

// Helper function to update Anonymous name in Excel
async function updateAnonymousNameInExcel(anonymousName, newName) {
  const workbook = new exceljs.Workbook();
  await workbook.xlsx.readFile(ATTENDANCE_FILE);
  
  const worksheet = workbook.getWorksheet(1);
  
  // Find the row with the anonymous name
  let anonymousRow = null;
  let rowNumber = 2; // Start from the second row (skip header)
  
  while (rowNumber <= worksheet.rowCount) {
    const row = worksheet.getRow(rowNumber);
    const name = row.getCell(1).value;
    
    if (name === anonymousName) {
      anonymousRow = row;
      break;
    }
    
    rowNumber++;
  }
  
  if (!anonymousRow) {
    throw new Error(`Anonymous name ${anonymousName} not found in the attendance sheet`);
  }
  
  // Update the name
  anonymousRow.getCell(1).value = newName;
  
  // Save the workbook
  await workbook.xlsx.writeFile(ATTENDANCE_FILE);
  
  return { success: true, id: rowNumber - 1 };
}

// API endpoints
// Get all attendance data
app.get('/api/attendance', async (req, res) => {
  try {
    const data = await readAttendanceData();
    res.json(data);
  } catch (error) {
    console.error('Error reading attendance data:', error);
    res.status(500).json({ error: 'Failed to read attendance data' });
  }
});

// Update attendance
app.put('/api/attendance', async (req, res) => {
  try {
    const { date, studentId, status, time } = req.body;
    const result = await updateAttendanceData(date, studentId, status, time);
    res.json(result);
  } catch (error) {
    console.error('Error updating attendance:', error);
    res.status(500).json({ error: 'Failed to update attendance' });
  }
});

// Add a new student with face encodings
app.post('/api/students', upload.any(), async (req, res) => {
  try {
    console.log('Request body:', req.body);
    console.log('Files:', req.files);
    
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    
    const name = req.body.name || 'Unknown Student';
    const uploadedFile = req.files[0];
    
    console.log(`Processing student registration for: ${name}`);
    
    // Step 1: Get face encodings from the uploaded image using the worker script
    const imagePath = uploadedFile.path;
    console.log(`Getting face encodings from: ${imagePath}`);
    const faceEncodings = await getFaceEncodings(imagePath);
    
    if (!faceEncodings || faceEncodings.length === 0) {
      console.log('No face detected in the image, deleting uploaded file');
      fs.unlinkSync(imagePath);
      return res.status(400).json({ error: 'No face detected in the image' });
    }
    
    console.log('Face encodings generated successfully');
    
    // Step 2: Check if this person matches any anonymous faces using the worker script
    let matchedAnonymous = null;
    let matchedAnonymousIndex = -1;
    
    if (fs.existsSync(ANONYMOUS_FILE)) {
      try {
        console.log('Checking for matching anonymous faces');
        // Save the face encoding to a temporary file for the worker
        const tempEncodingPath = path.join(TEMP_DIR, `temp_encoding_${Date.now()}.json`);
        fs.writeFileSync(tempEncodingPath, JSON.stringify({ encodings: [faceEncodings[0]] }));
        
        // Run the worker to match with anonymous faces
        const matchResult = await runFaceWorker('match_anonymous', [
          tempEncodingPath, 
          ANONYMOUS_FILE
        ]);
        
        // Clean up the temporary file
        fs.unlinkSync(tempEncodingPath);
        
        if (matchResult && matchResult.matched) {
          matchedAnonymous = matchResult.name;
          matchedAnonymousIndex = matchResult.index;
          console.log(`Matched anonymous face: ${matchedAnonymous} (index: ${matchedAnonymousIndex})`);
          
          // Update the anonymous student's name in Excel
          try {
            console.log(`Updating anonymous name ${matchedAnonymous} to ${name} in Excel`);
            const updateResult = await updateAnonymousNameInExcel(matchedAnonymous, name);
            
            // Remove this person from anonymous database using the worker
            if (matchedAnonymousIndex >= 0) {
              console.log(`Removing ${matchedAnonymous} from anonymous database`);
              const removeResult = await runFaceWorker('update_anonymous', [
                ANONYMOUS_FILE,
                matchedAnonymousIndex.toString()
              ]);
              
              if (removeResult && removeResult.success) {
                console.log(`Successfully removed ${removeResult.removed_name} from anonymous database`);
              } else {
                console.error('Failed to update anonymous database:', removeResult);
              }
            }
          } catch (excelError) {
            console.error('Excel error:', excelError);
            return res.status(500).json({ error: 'Excel file is currently in use. Please close it and try again.' });
          }
        } else {
          console.log('No matching anonymous face found');
        }
      } catch (error) {
        console.error('Error checking anonymous faces:', error);
      }
    }
    
    // Step 3: Add student to Excel if not already matched with anonymous
    let result;
    if (!matchedAnonymous) {
      try {
        // Check if the student name already exists in the Excel file
        console.log('Checking if student exists in Excel');
        const existingData = await readAttendanceData();
        const existingStudent = existingData.students.find(student => student.name === name);
        
        if (!existingStudent) {
          // Add new student to Excel
          console.log(`Adding new student ${name} to Excel`);
          result = await addNewStudentToExcel(name);
          console.log(`Added new student ${name} to Excel with ID ${result.id}`);
        } else {
          console.log(`Student ${name} already exists in Excel with ID ${existingStudent.id}`);
          result = { id: existingStudent.id };
        }
      } catch (excelError) {
        console.error('Excel error:', excelError);
        return res.status(500).json({ error: 'Excel file is currently in use. Please close it and try again.' });
      }
    } else {
      // For anonymous matches, get the ID from the update result
      result = { id: matchedAnonymousIndex }; // This is just a placeholder, should be the actual ID
    }
    
    // Step 4: Update the encodings database using the worker script
    const studentId = matchedAnonymous ? 
      result ? result.id.toString() : Date.now().toString() : 
      result ? result.id.toString() : Date.now().toString();
      
    console.log(`Updating student encodings for ${name} with ID ${studentId}`);
    
    // Save the face encoding to a temporary file for the worker
    const tempEncodingPath = path.join(TEMP_DIR, `temp_encoding_${Date.now()}.json`);
    fs.writeFileSync(tempEncodingPath, JSON.stringify({ encodings: [faceEncodings[0]] }));
    
    // Run the worker to update the student database
    const updateResult = await runFaceWorker('update_student', [
      KNOWN_ENCODINGS_FILE,
      studentId,
      name,
      tempEncodingPath,
      uploadedFile.path
    ]);
    
    // Clean up the temporary file
    fs.unlinkSync(tempEncodingPath);
    
    if (updateResult && updateResult.success) {
      console.log(`Successfully ${updateResult.action} student in encodings database`);
    } else {
      console.error('Failed to update student encodings:', updateResult);
    }
    
    console.log(`Student registration completed for: ${name}`);
    res.status(201).json({ 
      message: matchedAnonymous ? 
        'Anonymous student updated and registered successfully' : 
        'Student added successfully',
      id: studentId,
      name: name,
      photoPath: uploadedFile.filename // Return just the filename, not full path for security
    });
  } catch (error) {
    console.error('Error adding student:', error);
    res.status(500).json({ error: 'Failed to add student' });
  }
});

// Serve student photos
app.get('/api/photos/:filename', (req, res) => {
  const filename = req.params.filename;
  const filePath = path.join(STUDENT_IMAGES_FOLDER, filename);
  
  // Check if file exists
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).json({ error: 'Photo not found' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 