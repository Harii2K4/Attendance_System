import React, { useState, useEffect } from 'react';
import { updateAttendance } from '../services/api';
import '../styles/theme.css';

const AttendanceTable = ({ attendanceData, selectedDate, onAttendanceUpdated }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [students, setStudents] = useState([]);
  const [editingStudent, setEditingStudent] = useState(null);
  const [loadingStudentId, setLoadingStudentId] = useState(null);

  // Animation effect on mount
  useEffect(() => {
    setIsVisible(true);
  }, []);

  // Process attendance data when it changes
  useEffect(() => {
    if (!attendanceData || !selectedDate) return;
    
    const dateAttendance = attendanceData.attendance[selectedDate];
    if (!dateAttendance) return;

    // Create a list of students with their attendance status
    const studentsList = attendanceData.students.map(student => {
      const attendance = dateAttendance[student.id] || { status: 'Absent', time: '' };
      return {
        ...student,
        attendance
      };
    });

    setStudents(studentsList);
  }, [attendanceData, selectedDate]);

  // Function to handle attendance status changes
  const handleStatusChange = (studentId, newStatus) => {
    const updatedStudents = students.map(student => {
      if (student.id === studentId) {
        return {
          ...student,
          attendance: {
            ...student.attendance,
            status: newStatus
          }
        };
      }
      return student;
    });
    setStudents(updatedStudents);
  };

  // Function to handle time changes
  const handleTimeChange = (studentId, newTime) => {
    const updatedStudents = students.map(student => {
      if (student.id === studentId) {
        return {
          ...student,
          attendance: {
            ...student.attendance,
            time: newTime
          }
        };
      }
      return student;
    });
    setStudents(updatedStudents);
  };

  // Function to save attendance updates
  const handleSaveAttendance = async (studentId) => {
    const student = students.find(s => s.id === studentId);
    if (!student) return;

    setLoadingStudentId(studentId);
    
    try {
      await updateAttendance(
        selectedDate, 
        studentId, 
        student.attendance.status, 
        student.attendance.time
      );
      
      // Call the onAttendanceUpdated callback to refresh the parent component
      if (onAttendanceUpdated) {
        onAttendanceUpdated();
      }
      
      setEditingStudent(null);
    } catch (error) {
      console.error('Error updating attendance:', error);
    } finally {
      setLoadingStudentId(null);
    }
  };

  // Function to get status badge class
  const getStatusBadgeClass = (status) => {
    switch(status) {
      case 'Present': return 'badge badge-present';
      case 'Absent': return 'badge badge-absent';
      case 'Late': return 'badge badge-late';
      default: return 'badge';
    }
  };

  return (
    <div 
      className={`card ${isVisible ? 'slide-in-up' : ''}`}
      style={{ 
        margin: 'var(--spacing-md) 0',
        animationDelay: '0.2s',
        overflowX: 'auto'
      }}
    >
      <h2 style={{ 
        fontSize: '1.5rem', 
        marginTop: 0,
        marginBottom: 'var(--spacing-md)',
        color: 'var(--gray-800)'
      }}>
        Attendance Records
      </h2>
      
      <table className="table table-hover">
        <thead>
          <tr>
            <th>Student Name</th>
            <th>Status</th>
            <th>Time</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {students.map((student, index) => (
            <tr 
              key={student.id}
              className="slide-in-up"
              style={{ animationDelay: `${0.1 + (index * 0.05)}s` }}
            >
              <td>{student.name}</td>
              <td>
                {editingStudent === student.id ? (
                  <select 
                    className="select"
                    value={student.attendance.status}
                    onChange={(e) => handleStatusChange(student.id, e.target.value)}
                  >
                    <option value="Present">Present</option>
                    <option value="Absent">Absent</option>
                    <option value="Late">Late</option>
                  </select>
                ) : (
                  <span className={getStatusBadgeClass(student.attendance.status)}>
                    {student.attendance.status}
                  </span>
                )}
              </td>
              <td>
                {editingStudent === student.id ? (
                  <input 
                    type="time"
                    className="input"
                    value={student.attendance.time}
                    onChange={(e) => handleTimeChange(student.id, e.target.value)}
                    disabled={student.attendance.status === 'Absent'}
                  />
                ) : (
                  student.attendance.time || '-'
                )}
              </td>
              <td>
                {editingStudent === student.id ? (
                  <div style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
                    <button 
                      className="btn btn-success"
                      onClick={() => handleSaveAttendance(student.id)}
                      disabled={loadingStudentId === student.id}
                      style={{ minWidth: 'auto', padding: '0.25rem 0.5rem' }}
                    >
                      {loadingStudentId === student.id ? (
                        <svg 
                          className="spin" 
                          xmlns="http://www.w3.org/2000/svg" 
                          width="18" 
                          height="18" 
                          viewBox="0 0 24 24" 
                          fill="none" 
                          stroke="currentColor" 
                          strokeWidth="2" 
                          strokeLinecap="round" 
                          strokeLinejoin="round"
                          style={{ animation: 'spin 1s linear infinite' }}
                        >
                          <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                      )}
                    </button>
                    <button 
                      className="btn btn-danger"
                      onClick={() => setEditingStudent(null)}
                      style={{ minWidth: 'auto', padding: '0.25rem 0.5rem' }}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                      </svg>
                    </button>
                  </div>
                ) : (
                  <button 
                    className="btn btn-primary"
                    onClick={() => setEditingStudent(student.id)}
                    style={{ minWidth: 'auto', padding: '0.25rem 0.5rem' }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 20h9"></path>
                      <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                    </svg>
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AttendanceTable; 