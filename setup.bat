@echo off
echo ===================================================
echo     VisionAttend AI: Smart Attendance System Setup
echo ===================================================
echo [1/3] Creating local virtual environment (.venv)...
python -m venv .venv
echo [2/3] Upgrading pip and installing dependencies...
.\.venv\Scripts\pip install -r requirements.txt
echo [3/3] Setup complete!
echo ===================================================
echo Now double click 'run.bat' to start the application.
echo ===================================================
pause
