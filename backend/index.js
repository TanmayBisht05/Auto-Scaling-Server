/**
 * Purpose: Node.js target application. Simulates a standard microservice workload by combining asynchronous I/O wait times with synchronous CPU-bound calculations to evaluate autoscaler responsiveness.
 * Usage: Executed automatically by Docker Compose inside the backend containers.
 */

const express = require('express');
const app = express();
const PORT = 5000;

const BASE_PROCESSING_MS = 30;   // base async wait (simulates I/O)
const CPU_WORK_ITERATIONS = 50_000;  // lightweight CPU work per request

function doCpuWork() {
    // Lightweight synchronous work — fast enough not to block the event loop
    // but heavy enough to show meaningful CPU usage under load
    let result = 0;
    for (let i = 0; i < CPU_WORK_ITERATIONS; i++) {
        result += Math.sqrt(i);
    }
    return result;
}

app.get('/api', async (req, res) => {
    // 1. Simulate async I/O (non-blocking — other requests proceed during this)
    await new Promise(resolve =>
        setTimeout(resolve, BASE_PROCESSING_MS + Math.random() * 20)
    );

    // 2. Do lightweight CPU work
    const result = doCpuWork();

    res.json({
        status: 'ok',
        result: result,
        timestamp: Date.now()
    });
});

app.get('/health', (req, res) => res.json({ status: 'healthy' }));

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});