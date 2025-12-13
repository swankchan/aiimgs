# AI Image Search - 完整啟動指南

## 前置準備

### 1. Conda 環境設置

```bash
# 創建 conda 環境 (如果還沒有)
conda create -n aiimgs python=3.10

# 啟動環境
conda activate aiimgs
```

### 2. 安裝後端依賴

```bash
# 確保在 aiimgs 環境中
conda activate aiimgs

# 安裝 PyTorch (根據您的系統選擇)
# CPU 版本:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版本 (CUDA 11.8):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install -r requirements-api.txt
```

### 3. 設置數據庫

```bash
# 確保 PostgreSQL 正在運行

# 初始化數據庫和創建管理員用戶
python auth.py
```

### 4. 安裝前端依賴

```bash
cd frontend
npm install
cd ..
```

## 啟動應用

### 方法 1: 使用兩個終端

**終端 1 - 後端:**
```bash
# 啟動 conda 環境
conda activate aiimgs

# 啟動 FastAPI
uvicorn api.main:app --reload --port 8000
```

**終端 2 - 前端:**
```bash
cd frontend
npm run dev
```

### 方法 2: 使用啟動腳本

我為您創建了一個 PowerShell 腳本來自動啟動:

```bash
# 運行啟動腳本
.\START_DEV.ps1
```

## 訪問應用

- **前端**: http://localhost:3000
- **API 文檔**: http://localhost:8000/docs
- **後端健康檢查**: http://localhost:8000/api/health

## 默認登錄

- 用戶名: `admin`
- 密碼: `admin123`

⚠️ **首次登錄後請立即修改密碼!**

## 常見問題

### Q: 如何檢查 conda 環境是否啟動?
```bash
# 查看當前環境
conda info --envs

# 應該看到 aiimgs 環境前面有 * 號
```

### Q: PyTorch 安裝失敗?
根據您的系統選擇正確的版本:
- CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- GPU: 訪問 https://pytorch.org/get-started/locally/ 獲取適合您 CUDA 版本的命令

### Q: 數據庫連接失敗?
檢查 `db_config.py` 中的配置,確保 PostgreSQL 正在運行。

### Q: 前端無法連接後端?
確保:
1. 後端運行在 `http://localhost:8000`
2. 前端的 `next.config.js` 中的代理設置正確

### Q: 如何停止服務?
在兩個終端中按 `Ctrl+C`

## 開發提示

### 查看 API 文檔
FastAPI 自動生成 Swagger 文檔: http://localhost:8000/docs

### 熱重載
- 後端: 使用 `--reload` 參數,代碼修改後自動重啟
- 前端: Next.js 自動支持熱重載

### 查看日誌
後端和前端都會在終端輸出日誌,方便調試。

## Docker 部署 (生產環境)

```bash
# 構建和啟動所有服務
docker-compose -f docker-compose-v2.yml up -d

# 查看日誌
docker-compose -f docker-compose-v2.yml logs -f

# 停止服務
docker-compose -f docker-compose-v2.yml down
```

## 從舊版本遷移

如果您之前使用 Streamlit 版本:
1. 所有數據自動兼容,無需遷移
2. 圖像、PDF、索引文件保持原位
3. 數據庫結構相同
4. 配置文件 `config.json` 兼容

## 環境變量 (可選)

創建 `.env` 文件:
```env
# 數據庫
DATABASE_URL=postgresql://user:password@localhost:5432/aiimgs

# JWT 密鑰 (生產環境請修改)
SECRET_KEY=your-secret-key-here

# Ollama (如果使用 AI 分析)
OLLAMA_URL=http://localhost:11434
```

## 性能優化

### GPU 加速
如果有 NVIDIA GPU:
```bash
# 安裝 CUDA 版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CLIP 模型會自動使用 GPU
```

### 大量圖像索引
建議分批處理,每批 1000 張左右。

## 需要幫助?

查看詳細文檔:
- `README-v2.md` - 完整功能說明
- `AI_ANALYSIS_SETUP.md` - AI 分析設置
- `OLLAMA_SETUP.md` - Ollama 安裝指南
