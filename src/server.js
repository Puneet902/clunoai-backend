const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
require('dotenv').config();

const userRoutes = require('./routes/userRoutes');

const app = express();
const PORT = process.env.NODE_PORT || 5000;

// Security Middleware
app.use(helmet());
app.use(cors({
    origin: '*',
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Cluno-Dev']
}));
app.use(morgan('dev'));
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url} - Headers: ${JSON.stringify(req.headers['x-cluno-dev'])}`);
    next();
});
app.use(express.json({
    verify: (req, res, buf) => {
        req.rawBody = buf;
    }
}));

// Routes
app.use('/api/v1', userRoutes);

// Basic Health Check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date() });
});

// Centralized Error Handler
app.use((err, req, res, next) => {
    // Handle cases where err might be undefined or not an Error object
    const errorBody = err || new Error('Internal Server Error');
    console.error('Unhandled Error:', errorBody.stack || errorBody);

    const status = errorBody.status || 500;
    const message = process.env.NODE_ENV === 'production'
        ? 'An unexpected error occurred'
        : (errorBody.message || 'Unknown error');

    res.status(status).json({
        error: message,
        ...(process.env.NODE_ENV !== 'production' && { stack: errorBody.stack })
    });
});

const bridgeService = require('./services/bridgeService');

app.listen(PORT, () => {
    console.log(`🚀 Production Backend running on port ${PORT}`);
    bridgeService.start();
});
