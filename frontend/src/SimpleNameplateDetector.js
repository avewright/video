import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';

const SimpleNameplateDetector = () => {
  const webcamRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [lastDetection, setLastDetection] = useState(null);
  const [detectionCount, setDetectionCount] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  // Capture frame and send for detection
  const detectNameplate = async () => {
    if (!webcamRef.current || isProcessing) return;

    setIsProcessing(true);
    setError(null);

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        throw new Error('Failed to capture image from webcam');
      }

      // Convert base64 to blob
      const response = await fetch(imageSrc);
      const blob = await response.blob();

      // Create form data
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');
      formData.append('prompt', 'detect nameplate');

      // Send to API
      const apiResponse = await fetch('http://localhost:8000/inference', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`API request failed: ${apiResponse.status}`);
      }

      const result = await apiResponse.json();
      
      // Check if nameplate was detected
      const hasNameplate = result.success && result.raw_output && 
                          result.raw_output.toLowerCase().includes('detected');

      if (hasNameplate) {
        setDetectionCount(prev => prev + 1);
        setLastDetection({
          timestamp: new Date().toISOString(),
          image: imageSrc,
          result: result,
          confidence: result.confidence || 0.8
        });
      }

    } catch (err) {
      setError(err.message);
      console.error('Detection error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Auto-detection loop
  useEffect(() => {
    let interval;
    if (isDetecting) {
      interval = setInterval(detectNameplate, 1000); // Check every second
    }
    return () => clearInterval(interval);
  }, [isDetecting]);

  const extractFields = async () => {
    if (!lastDetection || isProcessing) return;

    setIsProcessing(true);
    setError(null);

    try {
      const response = await fetch(lastDetection.image);
      const blob = await response.blob();

      const formData = new FormData();
      formData.append('image', blob, 'nameplate.jpg');
      formData.append('prompt', 'Extract all key-value pairs from this nameplate image and return them in JSON format. Include information like model numbers, serial numbers, voltage, current, power, manufacturer, and any other technical specifications visible.');
      formData.append('max_new_tokens', '512');
      formData.append('temperature', '0.7');

      const apiResponse = await fetch('http://localhost:8000/inference', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`Field extraction failed: ${apiResponse.status}`);
      }

      const result = await apiResponse.json();
      
      setLastDetection(prev => ({
        ...prev,
        extractedFields: result.extracted_fields || result
      }));

    } catch (err) {
      setError(err.message);
      console.error('Field extraction error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h1>🏭 Simple Nameplate Detector</h1>
      
      <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
        <div style={{ flex: 1 }}>
          <h2>📹 Camera Feed</h2>
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <Webcam
              ref={webcamRef}
              audio={false}
              height={480}
              width={640}
              screenshotFormat="image/jpeg"
              style={{ border: '2px solid #ccc', borderRadius: '8px' }}
            />
            
            {isProcessing && (
              <div style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                background: 'rgba(0, 0, 0, 0.7)',
                color: 'white',
                padding: '5px 10px',
                borderRadius: '4px',
                fontSize: '12px'
              }}>
                Processing...
              </div>
            )}
            
            {lastDetection && (
                             <div style={{
                 position: 'absolute',
                 top: '10px',
                 left: '10px',
                 background: 'rgba(0, 255, 0, 0.8)',
                 color: 'white',
                 padding: '5px 10px',
                 borderRadius: '4px',
                 fontSize: '12px'
               }}>
                 🎯 Nameplate Detected! ({(lastDetection.confidence * 100).toFixed(1)}%)
               </div>
            )}
          </div>
          
          <div style={{ marginTop: '10px' }}>
            <button 
              onClick={() => setIsDetecting(!isDetecting)}
              style={{
                padding: '10px 20px',
                backgroundColor: isDetecting ? '#dc3545' : '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                marginRight: '10px'
              }}
            >
              {isDetecting ? '⏹️ Stop Detection' : '▶️ Start Detection'}
            </button>
            
            <button 
              onClick={detectNameplate}
              disabled={isProcessing}
              style={{
                padding: '10px 20px',
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isProcessing ? 'not-allowed' : 'pointer',
                marginRight: '10px'
              }}
            >
              📸 Detect Now
            </button>
            
            {lastDetection && (
              <button 
                onClick={extractFields}
                disabled={isProcessing}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#ffc107',
                  color: 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isProcessing ? 'not-allowed' : 'pointer'
                }}
              >
                🔍 Extract Fields
              </button>
            )}
          </div>
        </div>
        
        <div style={{ flex: 1 }}>
          <h2>📊 Results</h2>
          
          <div style={{ marginBottom: '20px' }}>
            <strong>Detection Count:</strong> {detectionCount}
          </div>
          
          {error && (
            <div style={{
              padding: '10px',
              backgroundColor: '#f8d7da',
              color: '#721c24',
              border: '1px solid #f5c6cb',
              borderRadius: '4px',
              marginBottom: '20px'
            }}>
              ❌ Error: {error}
            </div>
          )}
          
          {lastDetection && (
            <div style={{
              border: '1px solid #ccc',
              borderRadius: '8px',
              padding: '15px',
              marginBottom: '20px'
            }}>
              <h3>🎯 Last Detection</h3>
              <p><strong>Time:</strong> {new Date(lastDetection.timestamp).toLocaleString()}</p>
              
              <div style={{ marginTop: '10px' }}>
                <img 
                  src={lastDetection.image} 
                  alt="Detected nameplate" 
                  style={{ 
                    maxWidth: '200px', 
                    maxHeight: '150px',
                    border: '1px solid #ccc',
                    borderRadius: '4px'
                  }}
                />
              </div>
              
              {lastDetection.extractedFields && (
                <div style={{ marginTop: '15px' }}>
                  <h4>📋 Extracted Fields:</h4>
                  <pre style={{
                    backgroundColor: '#f8f9fa',
                    padding: '10px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    overflow: 'auto'
                  }}>
                    {JSON.stringify(lastDetection.extractedFields, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SimpleNameplateDetector; 