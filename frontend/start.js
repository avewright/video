#!/usr/bin/env node

/**
 * Startup script for Nameplate Detector Frontend
 * Checks prerequisites and starts services in the correct order
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting Nameplate Detector Application...');

// Start the backend server
console.log('📡 Starting backend server...');
const serverProcess = spawn('node', ['server/server.js'], {
  cwd: __dirname,
  stdio: 'inherit'
});

serverProcess.on('error', (error) => {
  console.error('❌ Backend server error:', error);
});

serverProcess.on('close', (code) => {
  console.log(`🔴 Backend server exited with code: ${code}`);
});

// Wait a bit for server to start, then start frontend
setTimeout(() => {
  console.log('⚛️  Starting React frontend...');
  const frontendProcess = spawn('npm', ['start'], {
    cwd: __dirname,
    stdio: 'inherit',
    shell: true
  });

  frontendProcess.on('error', (error) => {
    console.error('❌ Frontend error:', error);
  });

  frontendProcess.on('close', (code) => {
    console.log(`🔴 Frontend exited with code: ${code}`);
  });
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
    console.log('\n👋 Shutting down gracefully...');
    serverProcess.kill();
    frontendProcess.kill();
      process.exit(0);
    });
}, 3000);

// Instructions for the user
console.log(`
┌─────────────────────────────────────────────┐
│        🏭 Nameplate Detector Setup         │
├─────────────────────────────────────────────┤
│                                             │
│  1. Make sure your model API is running:   │
│     localhost:8000/inference                │
│                                             │
│  2. Backend server will start on:          │
│     localhost:3001                          │
│                                             │
│  3. Frontend will start on:                │
│     localhost:3000                          │
│                                             │
│  4. Access the app at:                      │
│     http://localhost:3000                   │
│                                             │
└─────────────────────────────────────────────┘
`);

// Keep the main process alive
process.stdin.resume(); 