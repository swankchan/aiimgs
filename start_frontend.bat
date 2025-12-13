@echo off
echo Starting AI Image Search Frontend...
echo.

REM Change to frontend directory
cd /d C:\AIIMGS\frontend

echo Starting Next.js development server...
echo Frontend will be available at http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

npm run dev
