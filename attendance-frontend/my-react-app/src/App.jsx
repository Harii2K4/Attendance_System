import { useState, useEffect, useCallback } from 'react'
import './App.css'
import './styles/theme.css'
import Header from './components/Header'
import Dashboard from './components/Dashboard'
import AttendanceTable from './components/AttendanceTable'
import AddStudentModal from './components/AddStudentModal'
import { fetchAttendanceData } from './services/api'

function App() {
  const [attendanceData, setAttendanceData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [showAddStudentModal, setShowAddStudentModal] = useState(false);

  // Memoize the loadAttendanceData function to prevent unnecessary recreations
  const loadAttendanceData = useCallback(async () => {
    setLoading(prevLoading => {
      // Only show loading indicator on initial load, not during refreshes
      if (attendanceData === null) return true;
      return prevLoading;
    });
    
    try {
      const data = await fetchAttendanceData();
      
      // Sort dates in descending order (latest date first)
      if (data && data.dates && data.dates.length > 0) {
        data.dates.sort((a, b) => new Date(b) - new Date(a));
      }
      
      setAttendanceData(data);
      
      // Set the first date (latest date) as the default selected date if no date is selected
      if (data && data.dates && data.dates.length > 0 && !selectedDate) {
        setSelectedDate(data.dates[0]);
      }
    } catch (err) {
      setError('Failed to load attendance data. Please try again.');
      console.error('Error loading attendance data:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedDate]);

  // Load attendance data on component mount
  useEffect(() => {
    loadAttendanceData();
  }, [loadAttendanceData]);
  
  // Set up polling to periodically refresh attendance data
  useEffect(() => {
    // Poll every 10 seconds for attendance updates
    const pollingInterval = setInterval(() => {
      console.log('Polling for attendance updates...');
      loadAttendanceData();
    }, 10000); // 10 seconds
    
    // Clean up polling interval when component unmounts
    return () => clearInterval(pollingInterval);
  }, [loadAttendanceData]);
  
  // Handler for manual refresh
  const handleRefresh = () => {
    console.log('Manual refresh triggered');
    loadAttendanceData();
  };

  return (
    <div className="app-container">
      <Header />
      
      {loading && !attendanceData ? (
        <div className="loading-container">
          <div className="spinner-container">
            <div className="spinner"></div>
          </div>
          <p>Loading attendance data...</p>
        </div>
      ) : error ? (
        <div className="error-container">
          <p>{error}</p>
          <button className="btn btn-primary" onClick={loadAttendanceData}>
            Try Again
          </button>
        </div>
      ) : attendanceData ? (
        <>
          <div className="action-bar">
            <div className="date-selector">
              <label htmlFor="date-select">Select Date:</label>
              <select 
                id="date-select"
                className="select"
                value={selectedDate || ''}
                onChange={(e) => setSelectedDate(e.target.value)}
              >
                {attendanceData.dates.map(date => (
                  <option key={date} value={date}>{date}</option>
                ))}
              </select>
            </div>
            
            <div className="action-buttons">
              <button 
                className="btn btn-secondary refresh-btn"
                onClick={handleRefresh}
                title="Refresh attendance data"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M23 4v6h-6"></path>
                  <path d="M1 20v-6h6"></path>
                  <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"></path>
                  <path d="M20.49 15a9 9 0 0 1-14.85 3.36L1 14"></path>
                </svg>
              </button>
              <button 
                className="btn btn-primary add-student-btn"
                onClick={() => setShowAddStudentModal(true)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                  <circle cx="8.5" cy="7" r="4"></circle>
                  <line x1="20" y1="8" x2="20" y2="14"></line>
                  <line x1="23" y1="11" x2="17" y2="11"></line>
                </svg>
                Add New Student
              </button>
            </div>
          </div>
          
          <Dashboard 
            attendanceData={attendanceData}
            selectedDate={selectedDate}
            loading={loading}
          />
          
          <AttendanceTable 
            attendanceData={attendanceData}
            selectedDate={selectedDate}
            onAttendanceUpdated={loadAttendanceData}
          />
          
          <AddStudentModal 
            isOpen={showAddStudentModal}
            onClose={() => setShowAddStudentModal(false)}
            onStudentAdded={loadAttendanceData}
          />
        </>
      ) : null}
    </div>
  );
}

export default App
