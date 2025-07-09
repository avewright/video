/**
 * Python Bridge Module
 * Handles communication between Node.js backend and Python nameplate detection service
 */

const { spawn } = require('child_process');
const path = require('path');

class PythonBridge {
  constructor() {
    this.pythonScriptPath = path.join(__dirname, '../python-integration/nameplate_service.py');
    this.modelPath = String.raw`C:\Users\AWright\OneDrive - Kahua, Inc\Projects\video\best_nameplate_classifier.pth`;
  }

  /**
   * Detect nameplate in base64 image data using Python service
   * @param {string} base64ImageData - Base64 encoded image data
   * @returns {Promise<object>} Detection result
   */
  detectNameplate(base64ImageData) {
    return new Promise((resolve, reject) => {
      const python = spawn('python', [this.pythonScriptPath]);
      
      let stdout = '';
      let stderr = '';
      
      // Send image data via stdin
      python.stdin.write(base64ImageData);
      python.stdin.end();
      
      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      python.on('close', (code) => {        
        if (code === 0) {
          try {
            const result = JSON.parse(stdout.trim());
            resolve(result);
          } catch (parseError) {
            reject(new Error(`Failed to parse Python output: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Python process exited with code ${code}: ${stderr}`));
        }
      });
      
      python.on('error', (error) => {
        reject(new Error(`Failed to spawn Python process: ${error.message}`));
      });
    });
  }

  /**
   * Check if Python service is available
   * @returns {Promise<boolean>} True if service is available
   */
  async checkAvailability() {
    try {
      // Test with a minimal base64 image (1x1 pixel)
      const testImageData = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAGA60e6kgAAAABJRU5ErkJggg==';
      const result = await this.detectNameplate(testImageData);
      return result.status === 'success' || result.status === 'error'; // Any response means service is available
    } catch (error) {
      return false;
    }
  }

  /**
   * Get model information
   * @returns {object} Model information
   */
  getModelInfo() {
    return {
      modelPath: this.modelPath,
      scriptPath: this.pythonScriptPath,
      modelExists: require('fs').existsSync(this.modelPath)
    };
  }
}

module.exports = PythonBridge; 