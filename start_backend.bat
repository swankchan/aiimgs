@echo off
echo Starting AI Image Search Backend...
echo.

REM Change to project directory
cd /d C:\AIIMGS

REM Activate conda environment and start backend
call conda activate aiimgs
if errorlevel 1 (
    echo Error: Failed to activate conda environment
    pause
    exit /b 1
)

echo Conda environment activated: aiimgs
echo Starting FastAPI server on http://localhost:8000
echo API docs will be available at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn api.main:app --reload --port 8000
