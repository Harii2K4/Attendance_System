// API service for attendance system
import axios from 'axios';

// Base URL for API requests - adjust this based on your backend server
const API_BASE_URL = 'http://localhost:5000/api';

// API functions
export const fetchAttendanceData = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/attendance`);
    return response.data;
  } catch (error) {
    console.error('Error fetching attendance data:', error);
    throw error;
  }
};

export const updateAttendance = async (date, studentId, status, time) => {
  try {
    const response = await axios.put(`${API_BASE_URL}/attendance`, {
      date,
      studentId,
      status,
      time
    });
    return response.data;
  } catch (error) {
    console.error('Error updating attendance:', error);
    throw error;
  }
};

export const addNewStudent = async (name, photoFile) => {
  // Create form data to send the file
  const formData = new FormData();
  formData.append('name', name);
  formData.append('photo', photoFile);
  
  try {
    const response = await axios.post(`${API_BASE_URL}/students`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error adding student:', error);
    throw error;
  }
}; 