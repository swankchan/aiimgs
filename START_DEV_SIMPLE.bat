@echo off
echo ========================================
echo   AI Image Search - 快速啟動
echo ========================================
echo.
echo 此腳本將在兩個新窗口中啟動服務:
echo   1. 後端 (FastAPI) - http://localhost:8000
echo   2. 前端 (Next.js) - http://localhost:3000
echo.
echo 默認登錄: admin / admin123
echo.
pause

REM 啟動後端
echo 正在啟動後端...
start "AI Image Search - Backend" cmd /k "conda activate aiimgs && uvicorn api.main:app --reload --port 8000"

REM 等待後端啟動
timeout /t 5 /nobreak

REM 啟動前端
echo 正在啟動前端...
start "AI Image Search - Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo   ✓ 服務啟動成功!
echo ========================================
echo.
echo 前端: http://localhost:3000
echo API 文檔: http://localhost:8000/docs
echo.
echo 提示: 關閉此窗口不會停止服務
echo       請關閉後端和前端窗口來停止服務
echo.
pause
