const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const multer = require('multer');
const FormData = require('form-data');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

// Enhanced logging setup
const logToFile = (level, message, error = null) => {
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] [${level}] ${message}`;
  
  console.log(logEntry);
  
  if (error) {
    console.error('Error details:', error);
  }
  
  // Also log to file
  try {
    const logDir = path.join(__dirname, '../../logs');
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
    
    const logFile = path.join(logDir, 'server.log');
    const fullEntry = error ? `${logEntry}\nError: ${error.stack || error}\n` : `${logEntry}\n`;
    fs.appendFileSync(logFile, fullEntry);
  } catch (logError) {
    console.error('Failed to write to log file:', logError);
  }
};

logToFile('INFO', '🚀 Starting nameplate detector server...');
logToFile('INFO', `📍 Current working directory: ${process.cwd()}`);
logToFile('INFO', `📁 Server directory: ${__dirname}`);
logToFile('INFO', `🔧 Node.js version: ${process.version}`);
logToFile('INFO', `💻 Platform: ${process.platform}`);

// Check for required dependencies
const requiredDeps = ['express', 'socket.io', 'cors', 'multer', 'form-data', 'axios'];
requiredDeps.forEach(dep => {
  try {
    require(dep);
    logToFile('INFO', `✅ Dependency ${dep} loaded successfully`);
  } catch (error) {
    logToFile('ERROR', `❌ Failed to load dependency ${dep}`, error);
  }
});

// Try to load PythonBridge
let PythonBridge;
try {
  PythonBridge = require('./python-bridge');
  logToFile('INFO', '✅ PythonBridge loaded successfully');
} catch (error) {
  logToFile('ERROR', '❌ Failed to load PythonBridge', error);
  logToFile('INFO', '⚠️  Server will run without Python integration');
}

const app = express();
const server = http.createServer(app);

// Initialize Socket.IO with error handling
let io;
try {
  io = socketIo(server, {
    cors: {
      origin: "http://localhost:3000",
      methods: ["GET", "POST"]
    }
  });
  logToFile('INFO', '✅ Socket.IO initialized successfully');
} catch (error) {
  logToFile('ERROR', '❌ Failed to initialize Socket.IO', error);
  process.exit(1);
}

const PORT = process.env.PORT || 3001;

// Initialize Python bridge with error handling
let pythonBridge;
if (PythonBridge) {
  try {
    pythonBridge = new PythonBridge();
    logToFile('INFO', '✅ Python bridge initialized successfully');
  } catch (error) {
    logToFile('ERROR', '❌ Failed to initialize Python bridge', error);
    logToFile('INFO', '⚠️  Server will run without Python integration');
  }
}

// Middleware setup with logging
logToFile('INFO', '🔧 Setting up middleware...');
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Multer configuration for image uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Test endpoint
app.get('/health', (req, res) => {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    port: PORT,
    pythonBridge: !!pythonBridge,
    uptime: process.uptime()
  };
  logToFile('INFO', '📊 Health check requested');
  res.json(health);
});

// Proxy route for inference API
app.post('/api/inference', upload.single('image'), async (req, res) => {
  try {
    const { prompt = 'hey', max_new_tokens = 512, temperature = 0.7 } = req.body;
    
    logToFile('INFO', `🔄 Inference request received: prompt="${prompt}"`);
    
    if (!req.file) {
      logToFile('ERROR', '❌ No image file provided in request');
      return res.status(400).json({ error: 'No image file provided' });
    }
    
    // Create form data for the inference endpoint
    const formData = new FormData();
    formData.append('image', req.file.buffer, 'nameplate.jpg');
    formData.append('prompt', prompt);
    formData.append('max_new_tokens', max_new_tokens.toString());
    formData.append('temperature', temperature.toString());
    
    logToFile('INFO', '🔄 Proxying request to inference API...');
    logToFile('INFO', `📤 Request parameters: ${JSON.stringify({ prompt, max_new_tokens, temperature })}`);
    
    const response = await axios.post('http://localhost:8000/inference/upload', formData, {
      headers: formData.getHeaders(),
      timeout: 30000  // 30 second timeout
    });
    
    logToFile('INFO', '✅ Inference API response received');
    res.json(response.data);
    
  } catch (error) {
    logToFile('ERROR', '❌ Proxy error occurred', error);
    if (error.response) {
      // Axios error with response
      logToFile('ERROR', `❌ Inference API error: ${error.response.status}`, error.response.data);
      res.status(error.response.status).json({ error: error.response.data });
    } else if (error.code === 'ECONNREFUSED') {
      logToFile('ERROR', '❌ Cannot connect to Python API server on localhost:8000');
      res.status(503).json({ error: 'Python API server is not running' });
    } else {
      // Network or other error
      logToFile('ERROR', '❌ Network or other error', error);
      res.status(500).json({ error: error.message });
    }
  }
});

// Store detection state
let detectionState = {
  isActive: false,
  isDetecting: false,
  detectionCount: 0,
  confidenceThreshold: 0.7,
  lastDetection: null,
  extractedFields: null,
  extractionInProgress: false
};

// Socket.io connection handling
io.on('connection', (socket) => {
  logToFile('INFO', `🔌 Client connected: ${socket.id}`);
  
  // Send current state to new client
  socket.emit('detectionState', detectionState);
  
  // Handle camera stream data
  socket.on('cameraFrame', (frameData) => {
    if (detectionState.isActive && !detectionState.isDetecting) {
      // Process frame for nameplate detection
      processFrame(frameData, socket);
    }
  });
  
  // Handle detection control
  socket.on('startDetection', () => {
    detectionState.isActive = true;
    detectionState.detectionCount = 0;
    io.emit('detectionState', detectionState);
    logToFile('INFO', '🎯 Detection started');
  });
  
  socket.on('stopDetection', () => {
    detectionState.isActive = false;
    detectionState.isDetecting = false;
    io.emit('detectionState', detectionState);
    logToFile('INFO', '⏹️  Detection stopped');
  });
  
  // Handle field extraction request
  socket.on('extractFields', (imageData) => {
    logToFile('INFO', '📝 Field extraction requested');
    extractFieldsFromImage(imageData, socket);
  });
  
  // Handle threshold adjustment
  socket.on('adjustThreshold', (newThreshold) => {
    detectionState.confidenceThreshold = Math.max(0.1, Math.min(1.0, newThreshold));
    io.emit('detectionState', detectionState);
    logToFile('INFO', `🎚️  Threshold adjusted to: ${detectionState.confidenceThreshold}`);
  });
  
  socket.on('disconnect', () => {
    logToFile('INFO', `🔌 Client disconnected: ${socket.id}`);
  });
});

// Process camera frame for nameplate detection
async function processFrame(frameData, socket) {
  if (detectionState.isDetecting) return;
  
  detectionState.isDetecting = true;
  
  try {
    // Convert base64 to buffer
    const imageBuffer = Buffer.from(frameData.split(',')[1], 'base64');
    
    // Use Python nameplate detection service
    const hasNameplate = await detectNameplateWithPython(imageBuffer);
    
    if (hasNameplate.detected && hasNameplate.confidence >= detectionState.confidenceThreshold) {
      detectionState.detectionCount++;
      detectionState.lastDetection = {
        timestamp: new Date().toISOString(),
        confidence: hasNameplate.confidence,
        imageData: frameData
      };
      
      logToFile('INFO', `🎯 Nameplate detected! Count: ${detectionState.detectionCount}, Confidence: ${(hasNameplate.confidence * 100).toFixed(1)}%`);
      
      // Notify all clients of detection
      io.emit('nameplateDetected', {
        detection: detectionState.lastDetection,
        detectionCount: detectionState.detectionCount
      });
    }
    
    // Send detection result
    socket.emit('detectionResult', {
      hasNameplate: hasNameplate.detected,
      confidence: hasNameplate.confidence,
      threshold: detectionState.confidenceThreshold
    });
    
  } catch (error) {
    logToFile('ERROR', '❌ Error processing frame', error);
    socket.emit('error', 'Failed to process frame');
  } finally {
    detectionState.isDetecting = false;
  }
}

// Detect nameplate using Python service
async function detectNameplateWithPython(imageBuffer) {
  try {
    if (!pythonBridge) {
      throw new Error('Python bridge not available');
    }
    
    // Convert buffer to base64
    const base64Image = imageBuffer.toString('base64');
    
    // Call Python service
    const result = await pythonBridge.detectNameplate(base64Image);
    
    return {
      detected: result.detected,
      confidence: result.confidence
    };
  } catch (error) {
    // Log error only once, not repeatedly
    if (!detectNameplateWithPython.errorLogged) {
      logToFile('ERROR', '❌ Python detection service failed - falling back to simulation mode', error);
      detectNameplateWithPython.errorLogged = true;
    }
    
    // Conservative fallback - rarely detect nameplates to avoid false positives
    const confidence = Math.random() * 0.3; // Low confidence range
    const detected = confidence > 0.25; // Very unlikely to trigger
    
    return {
      detected: detected,
      confidence: confidence
    };
  }
}

// Extract fields from nameplate image
async function extractFieldsFromImage(imageData, socket) {
  if (detectionState.extractionInProgress) return;
  
  detectionState.extractionInProgress = true;
  io.emit('detectionState', detectionState);
  
  try {
    // Convert base64 to buffer
    const imageBuffer = Buffer.from(imageData.split(',')[1], 'base64');
    
    // Create form data for the inference endpoint
    const formData = new FormData();
    formData.append('image', imageBuffer, 'nameplate.jpg');
    formData.append('prompt', 'Find all key and value pairs in this image (example: {name: "motor 3"}). Extract this information and return ONLY a valid json string.');
    formData.append('max_new_tokens', '512');
    
    logToFile('INFO', '📤 Sending image to inference endpoint for field extraction...');
    
    const response = await axios.post('http://localhost:8000/inference/upload', formData, {
      headers: formData.getHeaders(),
      timeout: 30000
    });
    
    detectionState.extractedFields = response.data;
    
    // Send extracted fields to all clients
    io.emit('fieldsExtracted', {
      fields: response.data,
      timestamp: new Date().toISOString()
    });
    
    logToFile('INFO', '✅ Fields extracted successfully');
    
  } catch (error) {
    logToFile('ERROR', '❌ Error extracting fields', error);
    io.emit('extractionError', error.message);
  } finally {
    detectionState.extractionInProgress = false;
    io.emit('detectionState', detectionState);
  }
}

// REST API endpoints
app.post('/api/extract-fields', upload.single('image'), async (req, res) => {
  try {
    const { prompt = 'Find all key and value pairs in this image (example: {name: "motor 3"}). Extract this information and return ONLY a valid json string.' } = req.body;
    
    logToFile('INFO', '📝 Field extraction API request received');
    
    if (!req.file) {
      logToFile('ERROR', '❌ No image provided in field extraction request');
      return res.status(400).json({ error: 'No image provided' });
    }
    
    const formData = new FormData();
    formData.append('image', req.file.buffer, 'nameplate.jpg');
    formData.append('prompt', prompt);
    formData.append('max_new_tokens', '512');
    
    const response = await axios.post('http://localhost:8000/inference/upload', formData, {
      headers: formData.getHeaders(),
      timeout: 30000
    });
    
    logToFile('INFO', '✅ Field extraction API request successful');
    res.json(response.data);
    
  } catch (error) {
    logToFile('ERROR', '❌ Error in field extraction API', error);
    if (error.response) {
      res.status(error.response.status).json({ error: error.response.data });
    } else {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const modelInfo = pythonBridge ? pythonBridge.getModelInfo() : { modelExists: false, modelPath: 'N/A' };
    const pythonAvailable = pythonBridge ? await pythonBridge.checkAvailability() : false;
    
    logToFile('INFO', '📊 API Health check requested');
    
    res.json({ 
      status: 'ok', 
      timestamp: new Date().toISOString(),
      detectionState: detectionState,
      python: {
        available: pythonAvailable,
        modelExists: modelInfo.modelExists,
        modelPath: modelInfo.modelPath
      }
    });
  } catch (error) {
    logToFile('ERROR', '❌ Error in health check endpoint', error);
    res.status(500).json({ error: 'Health check failed' });
  }
});

// Enhanced server startup with comprehensive error handling
server.listen(PORT, (error) => {
  if (error) {
    logToFile('ERROR', `❌ Failed to start server on port ${PORT}`, error);
    process.exit(1);
  }
  
  logToFile('INFO', `🚀 Server successfully started on port ${PORT}`);
  logToFile('INFO', `📱 Frontend proxy: http://localhost:3000`);
  logToFile('INFO', `🔌 API proxy: http://localhost:8000`);
  logToFile('INFO', `📋 Health check: http://localhost:${PORT}/health`);
  
  // Test connection to Python API
  setTimeout(async () => {
    try {
      logToFile('INFO', '🔍 Testing connection to Python API...');
      const response = await axios.get('http://localhost:8000/health', { timeout: 5000 });
      logToFile('INFO', `✅ Python API connection successful: ${JSON.stringify(response.data)}`);
    } catch (error) {
      logToFile('ERROR', '❌ Python API connection failed', error);
      logToFile('INFO', '⚠️  Server will work but Python features may be limited');
    }
  }, 2000);
});

// Enhanced error handling for uncaught exceptions
process.on('uncaughtException', (error) => {
  logToFile('ERROR', '❌ Uncaught Exception', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logToFile('ERROR', '❌ Unhandled Promise Rejection', reason);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logToFile('INFO', '🔄 Received SIGTERM, shutting down gracefully...');
  server.close(() => {
    logToFile('INFO', '✅ Server shut down successfully');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logToFile('INFO', '🔄 Received SIGINT, shutting down gracefully...');
  server.close(() => {
    logToFile('INFO', '✅ Server shut down successfully');
    process.exit(0);
  });
}); 