@echo off
echo ==========================================================
echo       Starting TrafficVioLens Auto-Setup ^& Runner
echo ==========================================================

:: Check for python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Check for node
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    pause
    exit /b 1
)

echo [1/4] Setting up Python virtual environment...
cd backend
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo [2/4] Installing Node dependencies...
cd frontend
call npm install
cd ..

echo [3/4] Starting backend server on http://localhost:8000...
cd backend
start /b cmd /c ".venv\Scripts\activate.bat && python -m uvicorn main:app --port 8000"
cd ..

echo [4/4] Starting React frontend server on http://localhost:5173...
cd frontend
npm run dev
