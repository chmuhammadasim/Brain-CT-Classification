import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState('');
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      setPreview(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult('');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(`Prediction: ${res.data.prediction} (Confidence: ${res.data.confidence})`);
    } catch (err) {
      setResult('Error: ' + err.message);
    }
    setLoading(false);
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Segoe UI, sans-serif'
    }}>
      <div style={{
        background: 'rgba(255,255,255,0.95)',
        padding: '2.8rem 2.2rem',
        borderRadius: '1.5rem',
        boxShadow: '0 12px 40px 0 rgba(60,72,100,0.18)',
        minWidth: 370,
        maxWidth: 95,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        border: '1px solid #e0e7ff',
        backdropFilter: 'blur(2px)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.7rem',
          marginBottom: '1.5rem'
        }}>
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="12" fill="#6366f1"/>
            <path d="M8 13l2.5 2.5L16 10" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <h2 style={{
            margin: 0,
            color: '#3730a3',
            fontWeight: 700,
            fontSize: '1.7rem',
            letterSpacing: '0.01em'
          }}>Image Classifier</h2>
        </div>
        <label htmlFor="file-upload" style={{
          background: 'linear-gradient(90deg, #6366f1 0%, #818cf8 100%)',
          color: '#fff',
          padding: '0.8rem 1.7rem',
          borderRadius: '0.7rem',
          cursor: 'pointer',
          marginBottom: '1.2rem',
          fontWeight: 600,
          fontSize: '1.05rem',
          boxShadow: '0 2px 8px rgba(99,102,241,0.10)',
          border: 'none',
          outline: 'none',
          transition: 'background 0.2s'
        }}>
          {file ? 'Change Image' : 'Choose Image'}
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </label>
        {preview && (
          <div style={{
            marginBottom: '1.1rem',
            borderRadius: '1rem',
            overflow: 'hidden',
            boxShadow: '0 4px 16px rgba(99,102,241,0.10)',
            border: '2.5px solid #a5b4fc'
          }}>
            <img
              src={preview}
              alt="Preview"
              style={{
                width: 200,
                height: 200,
                objectFit: 'cover',
                display: 'block',
                background: '#f1f5f9'
              }}
            />
          </div>
        )}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{
            background: file && !loading
              ? 'linear-gradient(90deg, #4f46e5 0%, #6366f1 100%)'
              : '#c7d2fe',
            color: file && !loading ? '#fff' : '#6366f1',
            padding: '0.7rem 1.5rem',
            border: 'none',
            borderRadius: '0.7rem',
            fontWeight: 600,
            fontSize: '1.05rem',
            cursor: file && !loading ? 'pointer' : 'not-allowed',
            marginBottom: '1.3rem',
            boxShadow: file && !loading ? '0 2px 8px rgba(99,102,241,0.10)' : 'none',
            transition: 'background 0.2s, color 0.2s'
          }}
        >
          {loading ? (
            <span>
              <span className="loader" style={{
                display: 'inline-block',
                width: 18,
                height: 18,
                border: '3px solid #fff',
                borderTop: '3px solid #6366f1',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
                marginRight: 8,
                verticalAlign: 'middle'
              }} />
              Uploading...
            </span>
          ) : 'Upload & Predict'}
        </button>
        <div style={{
          minHeight: '2.5rem',
          width: '100%',
          textAlign: 'center',
          color: result.startsWith('Error') ? '#dc2626' : (result ? '#059669' : '#64748b'),
          fontWeight: 500,
          fontSize: '1.13rem',
          background: result
            ? (result.startsWith('Error') ? '#fee2e2' : '#d1fae5')
            : 'transparent',
          borderRadius: '0.5rem',
          padding: result ? '0.7rem 0.5rem' : 0,
          marginTop: result ? '0.2rem' : 0,
          transition: 'background 0.2s'
        }}>
          {result || 'Prediction result will appear here.'}
        </div>
      </div>
      {/* Loader animation keyframes */}
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg);}
            100% { transform: rotate(360deg);}
          }
        `}
      </style>
    </div>
  );
}

export default App;
