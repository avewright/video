import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import io from 'socket.io-client';
import { 
  FaPlay, 
  FaPause, 
  FaStop, 
  FaCamera, 
  FaEye, 
  FaDownload,
  FaCog,
  FaSearch
} from 'react-icons/fa';
import './App.css';

const socket = io('http://localhost:3001');

function App() {
  const webcamRef = useRef(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionState, setDetectionState] = useState({
    isActive: false,
    isDetecting: false,
    detectionCount: 0,
    confidenceThreshold: 0.7,
    lastDetection: null,
    extractedFields: null,
    extractionInProgress: false
  });
  const [currentDetection, setCurrentDetection] = useState(null);
  const [extractedFields, setExtractedFields] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [streamActive, setStreamActive] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Add frame counter for logging
  const frameCounter = useRef(0);
  const lastLogTime = useRef(Date.now());

  // Log when component mounts
  useEffect(() => {
    console.log('🎬 Nameplate Detection App initializing...');
    console.log('📍 Connecting to backend server at http://localhost:3001');
    
    return () => {
      console.log('🔚 Nameplate Detection App unmounting...');
    };
  }, []);

  // Log webcam ready state
  const handleWebcamReady = () => {
    console.log('📷 Webcam ready and initialized');
    console.log('📊 Webcam element:', webcamRef.current ? 'Available' : 'Not available');
  };

  const handleWebcamError = (error) => {
    console.error('❌ Webcam error:', error);
  };

  useEffect(() => {
    // Socket event listeners
    socket.on('connect', () => {
      setIsConnected(true);
      console.log('🔌 Connected to server');
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
      console.log('🔌 Disconnected from server');
    });

    socket.on('detectionState', (state) => {
      console.log('📊 Detection state updated:', state);
      setDetectionState(state);
    });

    socket.on('nameplateDetected', (data) => {
      console.log('🎯 NAMEPLATE DETECTED!', data);
      console.log('📸 Detection data:', {
        timestamp: data.detection.timestamp,
        confidence: data.detection.confidence,
        detectionCount: data.detectionCount,
        imageDataLength: data.detection.imageData ? data.detection.imageData.length : 0
      });
      setCurrentDetection(data.detection);
    });

    socket.on('detectionResult', (result) => {
      // Handle real-time detection results
      console.log('🔍 Detection result received:', {
        hasNameplate: result.hasNameplate,
        confidence: result.confidence,
        threshold: result.threshold,
        timestamp: new Date().toISOString()
      });
      
      if (result.hasNameplate) {
        console.log('✅ Nameplate detected in frame - confidence:', (result.confidence * 100).toFixed(1) + '%');
      }
    });

    socket.on('error', (error) => {
      console.error('❌ Socket error:', error);
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('detectionState');
      socket.off('nameplateDetected');
      socket.off('detectionResult');
      socket.off('error');
    };
  }, []);

  // Send camera frames to server
  useEffect(() => {
    let intervalId;
    
    if (streamActive && webcamRef.current) {
      console.log('🎥 Starting camera frame capture - sending frame every 100ms');
      
      intervalId = setInterval(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          frameCounter.current++;
          
          // Log every 50 frames (5 seconds) to avoid spam
          if (frameCounter.current % 50 === 0) {
            const currentTime = Date.now();
            const timeSinceLastLog = currentTime - lastLogTime.current;
            const fps = (50 / timeSinceLastLog) * 1000;
            
            console.log(`📹 Frame ${frameCounter.current} sent to server`);
            console.log(`📊 Camera FPS: ${fps.toFixed(1)}`);
            console.log(`📏 Frame data length: ${imageSrc.length} characters`);
            console.log(`📸 Frame preview: ${imageSrc.substring(0, 50)}...`);
            
            lastLogTime.current = currentTime;
          }
          
          // Send frame to server for processing
          socket.emit('cameraFrame', imageSrc);
        } else {
          console.warn('⚠️ No image data from webcam');
        }
      }, 100); // Send frame every 100ms
    } else {
      if (streamActive) {
        console.warn('⚠️ Stream active but webcam not ready');
      }
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
        console.log('🛑 Camera frame capture stopped');
      }
    };
  }, [streamActive]);

  const startDetection = () => {
    console.log('🚀 Starting nameplate detection...');
    console.log('📋 Current detection state:', detectionState);
    
    frameCounter.current = 0;
    lastLogTime.current = Date.now();
    
    setStreamActive(true);
    socket.emit('startDetection');
    
    console.log('✅ Detection started - camera stream active');
  };

  const stopDetection = () => {
    console.log('🛑 Stopping nameplate detection...');
    
    setStreamActive(false);
    socket.emit('stopDetection');
    setCurrentDetection(null);
    setExtractedFields(null);
    
    console.log('✅ Detection stopped - camera stream inactive');
    console.log('📊 Total frames processed:', frameCounter.current);
  };

  const extractFields = async () => {
    console.log('🔄 extractFields function called');
    console.log('📸 currentDetection:', currentDetection);
    
    if (currentDetection && currentDetection.imageData) {
      try {
        console.log('✅ Have currentDetection and imageData');
        console.log('🖼️ Image data length:', currentDetection.imageData.length);
        console.log('🖼️ Image data preview:', currentDetection.imageData.substring(0, 50) + '...');
        
        // Set extraction in progress state
        setDetectionState(prev => ({
          ...prev,
          extractionInProgress: true
        }));
        console.log('⏳ Set extraction in progress to true');

        console.log('🔧 Starting image conversion...');

        // Convert base64 image to blob
        const base64Data = currentDetection.imageData.split(',')[1];
        console.log('📊 Base64 data length after split:', base64Data.length);
        
        const byteCharacters = atob(base64Data);
        console.log('📊 Byte characters length:', byteCharacters.length);
        
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        console.log('📊 Byte array length:', byteArray.length);
        
        const blob = new Blob([byteArray], { type: 'image/jpeg' });
        console.log('✅ Created blob - size:', blob.size, 'bytes, type:', blob.type);

        // Create FormData for the request
        console.log('📝 Creating FormData...');
        const formData = new FormData();
        formData.append('prompt', 'Find all key and value pairs in this image (example: {name: "motor 3"}). Extract this information and return ONLY a valid json string.');
        formData.append('image', blob, 'nameplate.jpg');
        formData.append('max_new_tokens', '512');

        console.log('📝 FormData created with:');
        console.log('  - prompt: "Find all key and value pairs in this image (example: {name: "motor 3"}). Extract this information and return ONLY a valid json string."');
        console.log('  - image: blob (', blob.size, 'bytes)');
        console.log('  - max_new_tokens: "512"');

        // Log all FormData entries
        for (let pair of formData.entries()) {
          if (pair[0] === 'image') {
            console.log('  - FormData entry:', pair[0], ':', '[Blob object]', pair[1].size, 'bytes');
          } else {
            console.log('  - FormData entry:', pair[0], ':', pair[1]);
          }
        }

        console.log('🚀 Making POST request to http://localhost:8000/inference/upload (direct to model API)');

        // Make the POST request directly to the model API
        const response = await fetch('http://localhost:8000/inference/upload', {
          method: 'POST',
          headers: {
            'accept': 'application/json',
          },
          body: formData,
        });

        console.log('📥 Response received!');
        console.log('📊 Response status:', response.status);
        console.log('📊 Response ok:', response.ok);
        console.log('📊 Response headers:');
        for (let pair of response.headers.entries()) {
          console.log('  -', pair[0], ':', pair[1]);
        }

        if (!response.ok) {
          console.error('❌ Response not ok, reading error text...');
          const errorText = await response.text();
          console.error('❌ Error response text:', errorText);
          throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        console.log('📖 Reading response JSON...');
        const result = await response.json();
        console.log('✅ API Response received:', result);
        console.log('📊 Response keys:', Object.keys(result));
        
        // Handle the new response format with success flag
        if (!result.success) {
          console.error('❌ API returned error:', result.error);
          throw new Error(`API Error: ${result.error}`);
        }
        
        // Handle the exact response schema from your API
        console.log('🔧 Setting extracted fields...');
        
        // Try to parse the JSON response from the model
        let parsedResponse = result.response;
        try {
          // If the model returned a JSON string, parse it
          if (typeof result.response === 'string') {
            // Clean up the response - remove markdown formatting if present
            let cleanResponse = result.response.trim();
            if (cleanResponse.startsWith('```json')) {
              cleanResponse = cleanResponse.replace(/```json\s*/, '').replace(/```\s*$/, '');
            }
            if (cleanResponse.startsWith('```')) {
              cleanResponse = cleanResponse.replace(/```[^`]*/, '').replace(/```\s*$/, '');
            }
            
            // Find JSON object in the response with better regex
            const jsonMatch = cleanResponse.match(/\{[\s\S]*?\}/);
            if (jsonMatch) {
              try {
                parsedResponse = JSON.parse(jsonMatch[0]);
                console.log('✅ Successfully parsed JSON response:', parsedResponse);
              } catch (parseError) {
                console.warn('⚠️ JSON parsing failed:', parseError.message);
                parsedResponse = { error: 'Malformed JSON structure', raw_response: result.response };
              }
            } else {
              console.warn('⚠️ No JSON object found in response');
              parsedResponse = { error: 'No valid JSON found', raw_response: result.response };
            }
          }
        } catch (e) {
          console.error('❌ Failed to parse JSON response:', e);
          parsedResponse = { error: 'Invalid JSON format', raw_response: result.response };
        }
        
        setExtractedFields({
          response: parsedResponse,
          raw_response: result.response,
          success: result.success,
          error: result.error
        });
        console.log('✅ Extracted fields set successfully');

      } catch (error) {
        console.error('💥 EXTRACTION ERROR:');
        console.error('🔴 Error object:', error);
        console.error('🔴 Error name:', error.name);
        console.error('🔴 Error message:', error.message);
        console.error('🔴 Error stack:', error.stack);
        
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
          console.error('🌐 This looks like a network/CORS error');
        }
        
        alert('Failed to extract fields: ' + error.message);
      } finally {
        console.log('🔄 Cleaning up - setting extraction in progress to false');
        // Reset extraction in progress state
        setDetectionState(prev => ({
          ...prev,
          extractionInProgress: false
        }));
      }
    } else {
      console.warn('⚠️ No currentDetection or imageData available');
      console.log('📊 currentDetection exists:', !!currentDetection);
      console.log('📊 imageData exists:', !!currentDetection?.imageData);
    }
  };

  const adjustThreshold = (newThreshold) => {
    socket.emit('adjustThreshold', newThreshold);
  };

  const downloadImage = () => {
    if (currentDetection && currentDetection.imageData) {
      const link = document.createElement('a');
      link.download = `nameplate_${new Date().toISOString()}.jpg`;
      link.href = currentDetection.imageData;
      link.click();
    }
  };

  const formatFields = (fields) => {
    if (!fields) return 'No fields extracted';
    
    // Handle the API response format
    if (fields.response) {
      return (
        <div className="response-content">
          <div className="field-item">
            <strong>📋 Extracted Nameplate Data:</strong>
            <div className="nameplate-data">
              {typeof fields.response === 'object' ? (
                fields.response.error ? (
                  // Display error message
                  <div className="error-response">
                    <div className="error-title">⚠️ Extraction Error</div>
                    <div className="error-message">{fields.response.error}</div>
                    {fields.response.raw_response && (
                      <div className="raw-response">
                        <strong>Raw Response:</strong>
                        <div className="response-text">{fields.response.raw_response}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  // Display JSON key-value pairs nicely
                  Object.entries(fields.response).map(([key, value]) => (
                    <div key={key} className="data-row">
                      <span className="data-key">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                      <span className="data-value">{value || 'N/A'}</span>
                    </div>
                  ))
                )
              ) : (
                // Display raw text response
                <div className="response-text">{fields.response}</div>
              )}
            </div>
          </div>
          <div className="field-item">
            <strong>🔧 Request Info:</strong>
            <div className="request-info">
              <div className="info-row">Has Image: {fields.has_image ? 'Yes' : 'No'}</div>
              <div className="info-row">Max Tokens: {fields.max_new_tokens}</div>
              <div className="info-row">Temperature: {fields.temperature}</div>
            </div>
          </div>
        </div>
      );
    }
    
    // Fallback for other formats
    if (typeof fields === 'string') {
      try {
        fields = JSON.parse(fields);
      } catch (e) {
        return fields;
      }
    }
    
    return Object.entries(fields).map(([key, value]) => (
      <div key={key} className="field-item">
        <strong>{key}:</strong> {value}
      </div>
    ));
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🏭 Nameplate Detector</h1>
        <div className="connection-status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
      </header>

      <div className="main-content">
        <div className="camera-section">
          <div className="camera-container">
            <Webcam
              ref={webcamRef}
              audio={false}
              height={480}
              width={640}
              screenshotFormat="image/jpeg"
              className="webcam"
              onUserMedia={handleWebcamReady}
              onError={handleWebcamError}
            />
            
            {currentDetection && (
              <div className="detection-overlay">
                <div className="detection-badge">
                  🎯 NAMEPLATE DETECTED!
                  <div className="confidence">
                    Confidence: {(currentDetection.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="controls">
            <button 
              className={`control-btn ${streamActive ? 'active' : ''}`}
              onClick={streamActive ? stopDetection : startDetection}
              disabled={!isConnected}
            >
              {streamActive ? (
                <>
                  <FaStop /> Stop Detection
                </>
              ) : (
                <>
                  <FaPlay /> Start Detection
                </>
              )}
            </button>

            <button 
              className="control-btn"
              onClick={() => setShowSettings(!showSettings)}
            >
              <FaCog /> Settings
            </button>

            {currentDetection && (
              <>
                <button 
                  className="control-btn extract-btn"
                  onClick={extractFields}
                  disabled={detectionState.extractionInProgress}
                >
                  <FaSearch /> 
                  {detectionState.extractionInProgress ? 'Extracting...' : 'Extract Fields'}
                </button>

                <button 
                  className="control-btn"
                  onClick={downloadImage}
                >
                  <FaDownload /> Download
                </button>
              </>
            )}
          </div>

          {showSettings && (
            <div className="settings-panel">
              <h3>Detection Settings</h3>
              <div className="setting-item">
                <label>Confidence Threshold: {(detectionState.confidenceThreshold * 100).toFixed(0)}%</label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={detectionState.confidenceThreshold * 100}
                  onChange={(e) => adjustThreshold(e.target.value / 100)}
                />
              </div>
            </div>
          )}
        </div>

        <div className="info-section">
          <div className="stats-panel">
            <h3>📊 Detection Stats</h3>
            <div className="stat-item">
              <span>Status:</span>
              <span className={`status ${streamActive ? 'active' : 'inactive'}`}>
                {streamActive ? '🟢 Active' : '🔴 Inactive'}
              </span>
            </div>
            <div className="stat-item">
              <span>Detections:</span>
              <span>{detectionState.detectionCount}</span>
            </div>
            <div className="stat-item">
              <span>Threshold:</span>
              <span>{(detectionState.confidenceThreshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          {currentDetection && (
            <div className="detection-panel">
              <h3>🎯 Latest Detection</h3>
              <div className="detection-info">
                <div className="detection-image">
                  <img 
                    src={currentDetection.imageData} 
                    alt="Detected nameplate" 
                    className="detected-image"
                  />
                </div>
                <div className="detection-details">
                  <div className="detail-item">
                    <strong>Confidence:</strong> {(currentDetection.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="detail-item">
                    <strong>Time:</strong> {new Date(currentDetection.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}

          {extractedFields && (
            <div className="fields-panel">
              <h3>📋 Extracted Fields</h3>
              <div className="fields-content">
                {formatFields(extractedFields)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App; 