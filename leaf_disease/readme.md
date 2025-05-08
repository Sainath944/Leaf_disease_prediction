# Plant Disease Detection System

A Flask-based web application that uses AI to detect plant diseases from leaf images and provides detailed information about the detected diseases.

## Features

- Upload and analyze plant leaf images
- AI-powered disease detection
- Detailed disease information including causes and prevention
- Interactive Q&A system about plant diseases

## Installation

1. Clone the repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
pip install google-generativeai
```

## Required Packages

- Flask
- Transformers
- PyTorch
- Pillow
- google-generativeai

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`
3. Upload a plant leaf image
4. View the disease detection results and recommendations

## Note

Make sure you have a valid Google AI API key configured in the application for the Q&A functionality to work.

## License

This project is open source and available under the MIT License.