import React, { useState, useRef } from 'react';
import { addNewStudent } from '../services/api';
import '../styles/theme.css';

const AddStudentModal = ({ isOpen, onClose, onStudentAdded }) => {
  const [studentName, setStudentName] = useState('');
  const [photo, setPhoto] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  
  // Reset form when modal opens/closes
  React.useEffect(() => {
    if (!isOpen) {
      setStudentName('');
      setPhoto(null);
      setPhotoPreview(null);
      setError('');
    }
  }, [isOpen]);
  
  // Handle photo selection
  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setError('Please upload a valid image file (JPG, PNG)');
      return;
    }
    
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPhotoPreview(reader.result);
    };
    reader.readAsDataURL(file);
    
    setPhoto(file);
    setError('');
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form
    if (!studentName.trim()) {
      setError('Please enter a student name');
      return;
    }
    
    if (!photo) {
      setError('Please upload a photo');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      await addNewStudent(studentName, photo);
      onStudentAdded();
      onClose();
    } catch (error) {
      setError('Failed to add student. Please try again.');
      console.error('Error adding student:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // If modal is closed, don't render anything
  if (!isOpen) return null;
  
  return (
    <div 
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        animation: 'fadeIn var(--transition-fast)'
      }}
    >
      <div 
        className="card slide-in-up"
        style={{ 
          width: '90%',
          maxWidth: '500px',
          maxHeight: '90vh',
          overflowY: 'auto',
          position: 'relative'
        }}
      >
        <button 
          onClick={onClose}
          style={{
            position: 'absolute',
            top: 'var(--spacing-md)',
            right: 'var(--spacing-md)',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--gray-500)',
            padding: 'var(--spacing-xs)',
            transition: 'color var(--transition-fast)',
            zIndex: 1
          }}
          className="btn"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
        
        <h2 style={{ 
          fontSize: '1.5rem', 
          marginTop: 0,
          marginBottom: 'var(--spacing-md)',
          color: 'var(--gray-800)'
        }}>
          Add New Student
        </h2>
        
        <form onSubmit={handleSubmit}>
          {/* Error message */}
          {error && (
            <div 
              style={{ 
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                color: 'var(--danger-color)',
                padding: 'var(--spacing-md)',
                borderRadius: 'var(--border-radius)',
                marginBottom: 'var(--spacing-md)'
              }}
            >
              {error}
            </div>
          )}
          
          {/* Student name input */}
          <div style={{ marginBottom: 'var(--spacing-md)' }}>
            <label 
              htmlFor="studentName" 
              style={{ 
                display: 'block', 
                marginBottom: 'var(--spacing-xs)',
                color: 'var(--gray-700)',
                fontWeight: 500
              }}
            >
              Student Name
            </label>
            <input
              id="studentName"
              type="text"
              className="input"
              value={studentName}
              onChange={(e) => setStudentName(e.target.value)}
              placeholder="Enter student name"
              disabled={isLoading}
            />
          </div>
          
          {/* Photo upload */}
          <div style={{ marginBottom: 'var(--spacing-lg)' }}>
            <label 
              htmlFor="studentPhoto" 
              style={{ 
                display: 'block', 
                marginBottom: 'var(--spacing-xs)',
                color: 'var(--gray-700)',
                fontWeight: 500
              }}
            >
              Student Photo
            </label>
            
            {/* Photo preview area */}
            <div 
              style={{ 
                width: '100%',
                height: '200px',
                border: '2px dashed var(--gray-300)',
                borderRadius: 'var(--border-radius)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 'var(--spacing-sm)',
                backgroundColor: 'var(--gray-100)',
                cursor: 'pointer',
                overflow: 'hidden'
              }}
              onClick={() => fileInputRef.current.click()}
            >
              {photoPreview ? (
                <img 
                  src={photoPreview} 
                  alt="Student preview" 
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '100%',
                    objectFit: 'cover'
                  }} 
                />
              ) : (
                <div style={{ textAlign: 'center', color: 'var(--gray-500)' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                  </svg>
                  <p style={{ margin: 'var(--spacing-sm) 0 0' }}>Click to upload photo</p>
                </div>
              )}
            </div>
            
            <input
              ref={fileInputRef}
              id="studentPhoto"
              type="file"
              accept="image/*"
              onChange={handlePhotoChange}
              style={{ display: 'none' }}
              disabled={isLoading}
            />
            
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => fileInputRef.current.click()}
              disabled={isLoading}
              style={{ width: '100%' }}
            >
              {photoPreview ? 'Change Photo' : 'Upload Photo'}
            </button>
          </div>
          
          {/* Submit button */}
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isLoading}
            style={{ width: '100%' }}
          >
            {isLoading ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 'var(--spacing-sm)' }}>
                <svg 
                  className="spin" 
                  xmlns="http://www.w3.org/2000/svg" 
                  width="20" 
                  height="20" 
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
                Adding Student...
              </div>
            ) : 'Add Student'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default AddStudentModal; 