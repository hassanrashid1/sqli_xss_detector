# SQL Injection & XSS Detection System

A Streamlit web application for detecting SQL Injection and XSS attacks using a deep learning model with cross-attention mechanism.

## Features

- 🛡️ Real-time security threat detection
- 📊 Visual probability breakdown
- 🎯 High accuracy (~99.4%)
- 🚀 Easy-to-use web interface
- 📈 Detailed prediction analysis

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure the model file `sqli_xss_detection_model.h5` is in the same directory as `app.py`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Usage

1. Enter a payload or query string in the text area
2. Click "Analyze Payload" button
3. View the prediction results with confidence scores
4. Check the probability breakdown for all classes

## Model Architecture

- **Dual Input**: Text characters + Special symbols
- **CNN + Bidirectional GRU**: For feature extraction
- **Cross-Attention**: To capture interactions between text and symbols
- **3-Class Classification**: SQL Injection, XSS, or Normal

## Example Payloads

- **SQL Injection**: `admin' OR '1'='1'--`
- **XSS**: `<script>alert('XSS')</script>`
- **Normal**: `SELECT name FROM users WHERE id = 123`

