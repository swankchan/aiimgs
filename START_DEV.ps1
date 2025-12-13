# AI Image Search - 快速啟動指南
# 
# 此腳本將自動啟動後端和前端服務
# 使用方法: .\START_DEV.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI Image Search - 啟動開發環境" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 檢查 conda 是否安裝
Write-Host "檢查 Conda 環境..." -ForegroundColor Yellow
$condaExists = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaExists) {
    Write-Host "❌ 錯誤: 未找到 conda 命令" -ForegroundColor Red
    Write-Host "請先安裝 Anaconda 或 Miniconda" -ForegroundColor Red
    exit 1
}

# 檢查 aiimgs 環境是否存在
Write-Host "檢查 aiimgs conda 環境..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "aiimgs"
if (-not $envExists) {
    Write-Host "⚠️  未找到 aiimgs 環境" -ForegroundColor Yellow
    Write-Host "正在創建 aiimgs conda 環境..." -ForegroundColor Yellow
    conda create -n aiimgs python=3.10 -y
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 創建環境失敗" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Conda 環境就緒" -ForegroundColor Green
Write-Host ""

# 檢查後端依賴
Write-Host "檢查後端依賴..." -ForegroundColor Yellow
$requirementsPath = "requirements-api.txt"
if (-not (Test-Path $requirementsPath)) {
    Write-Host "❌ 未找到 requirements-api.txt" -ForegroundColor Red
    exit 1
}

# 檢查前端依賴
Write-Host "檢查前端依賴..." -ForegroundColor Yellow
if (-not (Test-Path "frontend/node_modules")) {
    Write-Host "⚠️  前端依賴未安裝,正在安裝..." -ForegroundColor Yellow
    Push-Location frontend
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ 前端依賴安裝失敗" -ForegroundColor Red
        Pop-Location
        exit 1
    }
    Pop-Location
    Write-Host "✓ 前端依賴安裝完成" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  啟動服務" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📝 提示:" -ForegroundColor Yellow
Write-Host "  - 後端將運行在: http://localhost:8000" -ForegroundColor White
Write-Host "  - 前端將運行在: http://localhost:3000" -ForegroundColor White
Write-Host "  - API 文檔: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  默認登錄:" -ForegroundColor White
Write-Host "  用戶名: admin" -ForegroundColor White
Write-Host "  密碼: admin123" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  按 Ctrl+C 停止服務" -ForegroundColor Yellow
Write-Host ""

# 啟動後端 (使用 conda run)
Write-Host "🚀 啟動後端..." -ForegroundColor Green
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & conda run -n aiimgs uvicorn api.main:app --reload --port 8000
}

# 等待後端啟動
Write-Host "等待後端啟動..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 啟動前端
Write-Host "🚀 啟動前端..." -ForegroundColor Green
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD/frontend
    npm run dev
}

# 等待前端啟動
Write-Host "等待前端啟動..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✓ 服務啟動成功!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "訪問應用: http://localhost:3000" -ForegroundColor Cyan
Write-Host "API 文檔: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "查看日誌:" -ForegroundColor Yellow
Write-Host "  後端日誌: Receive-Job -Id $($backendJob.Id) -Keep" -ForegroundColor White
Write-Host "  前端日誌: Receive-Job -Id $($frontendJob.Id) -Keep" -ForegroundColor White
Write-Host ""
Write-Host "按 Enter 查看實時日誌..." -ForegroundColor Yellow
Read-Host

# 顯示實時日誌
try {
    while ($true) {
        Clear-Host
        Write-Host "========== 後端日誌 ==========" -ForegroundColor Cyan
        Receive-Job -Id $backendJob.Id -Keep | Select-Object -Last 10
        Write-Host ""
        Write-Host "========== 前端日誌 ==========" -ForegroundColor Cyan
        Receive-Job -Id $frontendJob.Id -Keep | Select-Object -Last 10
        Write-Host ""
        Write-Host "按 Ctrl+C 停止服務" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
}
finally {
    # 清理
    Write-Host ""
    Write-Host "正在停止服務..." -ForegroundColor Yellow
    Stop-Job -Id $backendJob.Id
    Stop-Job -Id $frontendJob.Id
    Remove-Job -Id $backendJob.Id
    Remove-Job -Id $frontendJob.Id
    Write-Host "✓ 服務已停止" -ForegroundColor Green
}
