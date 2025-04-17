import React, { useState, useEffect, useRef } from 'react';
import '../styles/theme.css';

const Dashboard = ({ attendanceData, selectedDate, loading = false }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [metrics, setMetrics] = useState({
    total: 0,
    present: 0,
    absent: 0,
    late: 0
  });
  
  // Keep previous metrics for animation
  const prevMetricsRef = useRef(metrics);

  // Animation effect on mount
  useEffect(() => {
    setIsVisible(true);
  }, []);
  
  // Handle refresh animation when loading state changes
  useEffect(() => {
    if (loading) {
      setRefreshing(true);
    } else {
      // Keep refreshing state for a moment to complete animation
      const timer = setTimeout(() => {
        setRefreshing(false);
      }, 600);
      return () => clearTimeout(timer);
    }
  }, [loading]);

  // Calculate metrics when data or selected date changes
  useEffect(() => {
    if (!attendanceData || !selectedDate) return;

    const dateAttendance = attendanceData.attendance[selectedDate];
    if (!dateAttendance) return;
    
    // Save previous metrics for animation
    prevMetricsRef.current = metrics;

    const studentCount = attendanceData.students.length;
    let presentCount = 0;
    let absentCount = 0;
    let lateCount = 0;

    // Count each status
    Object.values(dateAttendance).forEach(entry => {
      if (entry.status === 'Present') presentCount++;
      else if (entry.status === 'Late') lateCount++;
      else absentCount++;
    });

    setMetrics({
      total: studentCount,
      present: presentCount,
      absent: absentCount,
      late: lateCount
    });
  }, [attendanceData, selectedDate]);

  return (
    <div 
      className={`card dashboard-card ${isVisible ? 'slide-in-up' : ''}`}
      style={{ 
        margin: 'var(--spacing-md) 0',
        animationDelay: '0.1s',
        position: 'relative',
        overflow: 'hidden',
        borderRadius: 'var(--border-radius-lg)',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)'
      }}
    >
      {refreshing && (
        <div className="refresh-indicator" style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '3px',
          background: 'linear-gradient(90deg, #3a86ff, #ff006e)',
          backgroundSize: '200% 100%',
          animation: 'gradient-slide 2s linear infinite, loading-bar 1s infinite linear'
        }}></div>
      )}
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ 
          fontSize: '1.5rem', 
          marginTop: 0,
          marginBottom: 'var(--spacing-md)',
          color: 'var(--gray-800)',
          fontWeight: '600'
        }}>
          Attendance Overview
        </h2>
        
        {loading && (
          <div className="pulse" style={{ 
            fontSize: '0.8rem', 
            color: 'var(--primary-color)',
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            fontWeight: '500'
          }}>
            <span className="dot-pulse"></span>
            Updating
          </div>
        )}
      </div>
      
      <div className="grid grid-cols-4">
        {/* Total Students */}
        <MetricCard 
          title="Total Students"
          value={metrics.total}
          prevValue={prevMetricsRef.current.total}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
              <circle cx="9" cy="7" r="4"></circle>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
          }
          color="var(--primary-color)"
          delay={0.2}
          refreshing={refreshing}
        />
        
        {/* Present */}
        <MetricCard 
          title="Present"
          value={metrics.present}
          prevValue={prevMetricsRef.current.present}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
              <polyline points="22 4 12 14.01 9 11.01"></polyline>
            </svg>
          }
          color="var(--success-color)"
          delay={0.3}
          refreshing={refreshing}
        />
        
        {/* Absent */}
        <MetricCard 
          title="Absent"
          value={metrics.absent}
          prevValue={prevMetricsRef.current.absent}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="15" y1="9" x2="9" y2="15"></line>
              <line x1="9" y1="9" x2="15" y2="15"></line>
            </svg>
          }
          color="var(--danger-color)"
          delay={0.4}
          refreshing={refreshing}
        />
        
        {/* Late */}
        <MetricCard 
          title="Late"
          value={metrics.late}
          prevValue={prevMetricsRef.current.late}
          icon={
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <polyline points="12 6 12 12 16 14"></polyline>
            </svg>
          }
          color="var(--warning-color)"
          delay={0.5}
          refreshing={refreshing}
        />
      </div>
      
      <style jsx>{`
        @keyframes loading-bar {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        @keyframes gradient-slide {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        .dot-pulse {
          position: relative;
          width: 10px;
          height: 10px;
          border-radius: 5px;
          background-color: var(--primary-color);
          animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
          0% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.1); }
          100% { opacity: 0.3; transform: scale(0.8); }
        }
        
        .value-change-up {
          animation: highlight-up 1s ease-out;
        }
        
        .value-change-down {
          animation: highlight-down 1s ease-out;
        }
        
        @keyframes highlight-up {
          0% { color: inherit; transform: scale(1); }
          50% { color: var(--success-color); transform: scale(1.1); }
          100% { color: inherit; transform: scale(1); }
        }
        
        @keyframes highlight-down {
          0% { color: inherit; transform: scale(1); }
          50% { color: var(--danger-color); transform: scale(1.1); }
          100% { color: inherit; transform: scale(1); }
        }
      `}</style>
    </div>
  );
};

// Metric card component for dashboard
const MetricCard = ({ title, value, prevValue, icon, color, delay, refreshing }) => {
  // Determine if value has changed for animation
  const valueChanged = prevValue !== undefined && value !== prevValue;
  const valueIncreased = valueChanged && value > prevValue;
  const valueDecreased = valueChanged && value < prevValue;
  
  return (
    <div 
      className={`card slide-in-up ${refreshing ? 'card-refresh' : ''}`} 
      style={{ 
        padding: 'var(--spacing-md)',
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--spacing-md)',
        animationDelay: `${delay}s`,
        boxShadow: `0 4px 6px rgba(0, 0, 0, 0.05), 0 0 0 3px ${color}10`,
        border: `1px solid ${color}30`,
        transition: 'transform 0.3s ease, box-shadow 0.3s ease',
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        backdropFilter: 'blur(10px)',
        borderRadius: '12px'
      }}
    >
      <div style={{ 
        width: '48px', 
        height: '48px',
        borderRadius: '50%',
        backgroundColor: `${color}15`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: color,
      }}>
        {icon}
      </div>
      <div>
        <h3 style={{ 
          fontSize: '0.85rem', 
          fontWeight: '500',
          color: 'var(--gray-600)',
          margin: 0,
          marginBottom: 'var(--spacing-xs)'
        }}>
          {title}
        </h3>
        <p 
          className={`
            ${valueIncreased ? 'value-change-up' : ''}
            ${valueDecreased ? 'value-change-down' : ''}
          `}
          style={{ 
            fontSize: '1.75rem', 
            fontWeight: '700',
            margin: 0,
            color: color,
            display: 'flex',
            alignItems: 'center',
            gap: '5px'
          }}
        >
          {value}
          {valueChanged && (
            <span style={{ 
              fontSize: '0.8rem', 
              fontWeight: 'normal', 
              color: valueIncreased ? 'var(--success-color)' : 'var(--danger-color)',
              marginTop: '2px'
            }}>
              {valueIncreased ? '↑' : '↓'} 
              {Math.abs(value - prevValue)}
            </span>
          )}
        </p>
      </div>
      
      <style jsx>{`
        .card-refresh {
          animation: card-pulse 1s ease-in-out;
        }
        
        @keyframes card-pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.02); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15), 0 0 0 3px ${color}30; }
          100% { transform: scale(1); }
        }
        
        [data-theme="dark"] & {
          background-color: rgba(30, 30, 30, 0.7);
          color: var(--gray-300);
        }
      `}</style>
    </div>
  );
};

export default Dashboard; 