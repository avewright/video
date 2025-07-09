# Field Extraction Feature

## Overview

The nameplate detection system now includes automatic field extraction capabilities. When a nameplate is detected, the system can automatically extract key-value pairs from the nameplate image using an inference endpoint.

## Features

- **Automatic Detection**: When a nameplate is detected with sufficient confidence, the system pauses and prompts for field extraction
- **Manual Extraction**: Users can manually trigger field extraction when paused using the 'E' key
- **JSON Output**: Extracted fields are returned in JSON format with key-value pairs
- **Visual Feedback**: The system provides real-time feedback about extraction status

## Setup

### Prerequisites

1. **Inference Server**: You need a running inference server at `http://localhost:8000/inference`
2. **Dependencies**: Install the required packages:
   ```bash
   pip install requests>=2.31.0
   ```

### Inference Endpoint

The system expects an inference endpoint with the following specification:

```bash
curl -X 'POST' \
  'http://localhost:8000/inference' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'prompt=Extract all key-value pairs from this nameplate image and return them in JSON format' \
  -F 'image=@nameplate.jpg' \
  -F 'max_new_tokens=512' \
  -F 'temperature=0.7'
```

## Usage

### 1. Real-time Detection with Field Extraction

Run the nameplate detector:

```bash
python realtime_nameplate_detector.py
```

**Controls:**
- **SPACE**: Pause/Resume detection
- **E**: Extract fields (when paused)
- **Q**: Quit
- **O**: Toggle overlay
- **+/-**: Adjust confidence threshold
- **R**: Reset counters

### 2. Windows-Compatible Version

For Windows users experiencing OpenCV GUI issues:

```bash
python camera_detector_windows.py
```

**Controls:**
- **Close window**: Quit
- **SPACE**: Pause/Resume detection
- **E**: Extract fields (when paused)
- **+/-**: Adjust confidence threshold
- **R**: Reset counters

### 3. Launcher Script

Use the launcher to choose the best detection method:

```bash
python run_nameplate_detector.py
```

## How It Works

1. **Detection**: The system continuously scans for nameplates
2. **Auto-Pause**: When a nameplate is detected with sufficient confidence, the system pauses
3. **User Prompt**: The system prompts: "Would you like to extract fields from this nameplate? (y/n)"
4. **Field Extraction**: If you answer 'yes', the image is sent to the inference endpoint
5. **Results Display**: Extracted fields are displayed in JSON format in the console

## Example Output

```json
{
  "manufacturer": "Siemens",
  "model": "1LA7 090-4AA60",
  "serial_number": "ABC123456",
  "voltage": "400V",
  "current": "15.4A",
  "power": "4kW",
  "frequency": "50Hz",
  "rpm": "1450",
  "protection_class": "IP55"
}
```

## Testing

Test the field extraction functionality:

```bash
python test_field_extraction.py
```

This script will:
1. Check if the inference endpoint is available
2. Let you choose a test image
3. Send the image for field extraction
4. Display the results

## Configuration

### Confidence Threshold

Adjust the confidence threshold for automatic pausing:

```bash
python realtime_nameplate_detector.py --threshold 0.8
```

### Inference Endpoint URL

To use a different inference endpoint, modify the URL in the code:

```python
url = 'http://your-server:8000/inference'
```

### Custom Prompt

Modify the extraction prompt in the code:

```python
data = {
    'prompt': 'Your custom extraction prompt here',
    'max_new_tokens': 512,
    'temperature': 0.7
}
```

## Troubleshooting

### Common Issues

1. **Inference Server Not Available**
   ```
   ❌ Inference endpoint is not available at http://localhost:8000
   ```
   - Solution: Start your inference server first
   - Check if the server is running on the correct port

2. **Network Timeout**
   ```
   ❌ Network error during field extraction: timeout
   ```
   - Solution: Increase timeout in the code (default: 30 seconds)
   - Check your network connection

3. **API Request Failed**
   ```
   ❌ API request failed with status 500
   ```
   - Solution: Check inference server logs
   - Verify the image format and request parameters

### Debug Mode

For debugging, use the test script:

```bash
python test_field_extraction.py
```

This will test the endpoint connection and image processing without the camera interface.

## API Response Format

The inference endpoint should return JSON in one of these formats:

### Success Response
```json
{
  "status": "success",
  "extracted_fields": {
    "manufacturer": "Siemens",
    "model": "1LA7 090-4AA60",
    "voltage": "400V"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Could not extract fields from image"
}
```

## Performance Notes

- Field extraction adds ~1-5 seconds per request depending on the inference server
- The system uses threading to prevent UI freezing during extraction
- Images are sent as JPEG format to minimize transfer time
- Default timeout is 30 seconds

## Integration Tips

1. **Custom Prompts**: Modify the prompt to extract specific fields relevant to your use case
2. **Post-Processing**: Add validation and formatting of extracted fields
3. **Database Storage**: Save extracted fields to a database for tracking
4. **Batch Processing**: Process multiple detections in batch mode

## Security Considerations

- The system sends images over HTTP (not HTTPS by default)
- Consider using HTTPS for production environments
- Validate and sanitize extracted data before storage
- Implement rate limiting to prevent abuse

---

For more information, see the main documentation or contact support. 