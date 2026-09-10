# VisionAttend AI - Enterprise Biometric Facial Attendance System

VisionAttend AI is an enterprise-grade biometric facial recognition attendance platform designed for academic institutions and modern workplaces. Powered by Deep Learning (MobileNetV2 Transfer Learning) and LiteRT / TensorFlow, the system provides real-time optical face detection, anti-spoofing liveness guards, interactive digital attendance sheets, and an immersive entrance kiosk terminal.

## Key Features

- **Real-Time Facial Recognition**: Sub-15ms inference latency powered by LiteRT (TensorFlow Lite) XNNPACK CPU acceleration.
- **Anti-Spoofing & Liveness Guard**: Dual-layer eye verification and Laplacian texture variance checking to prevent printed photo and screen replay attacks.
- **Immersive Classroom Kiosk Terminal**: Responsive, full-viewport biometric kiosk interface (`/kiosk`) with optical targeting reticles, scanning lasers, and instant audio feedback.
- **Interactive Attendance Sheet**: Dynamic date-filtered attendance table with manual override actions (`Present`, `Late`, `Absent`) and one-click bulk marking.
- **Dataset Creation & Colab Integration**: Built-in 60-sample facial dataset capture tool (`/students`) with 1-click `dataset.zip` export for automated Google Colab retraining.
- **Dynamic System Configuration**: Customizable confidence match threshold (e.g. 80%), auto-absent cutoff schedules, and institution branding persisted in SQLite.
- **Analytics & Reporting**: Real-time KPI dashboards, attendance rate graphs, and Excel/CSV export capabilities.

## Technology Stack

- **Backend**: Python 3.10+, Flask, Flask-SQLAlchemy, Gunicorn
- **Computer Vision & AI**: OpenCV, MobileNetV2 (Transfer Learning), LiteRT / TensorFlow Lite, Haar Cascades
- **Database**: SQLite (SQLAlchemy ORM)
- **Frontend**: Tailwind CSS, Font Awesome 6, Chart.js, HTML5 Canvas / WebRTC

## Quick Start

```bash
# Clone the repository
git clone https://github.com/HamzaRashid21/visionattendence-ai.git
cd visionattendence-ai

# Create virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Launch application
python app.py
```

Access the application in your browser:
- Admin Dashboard: `http://localhost:5000` (Default: `admin` / `admin123`)
- Kiosk Terminal: `http://localhost:5000/kiosk` (Public, no login required)
