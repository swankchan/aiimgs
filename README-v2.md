# AI Image Search - Next.js + FastAPI

完整的 CLIP 圖像搜索應用,使用 Next.js (React) 前端和 FastAPI 後端。

## 架構

- **前端**: Next.js 14 + React + TypeScript + Tailwind CSS
- **後端**: FastAPI + Python
- **數據庫**: PostgreSQL
- **AI 模型**: OpenAI CLIP (ViT-B/16)
- **向量搜索**: FAISS

## 功能特性

### 🔍 搜索功能
- 文本搜索: 使用自然語言描述查找圖像
- 圖像搜索: 上傳圖像查找相似圖像
- 實時搜索結果與相似度評分
- 分頁顯示結果

### 📚 索引管理
- 批量圖像索引
- 同步多個文件夾
- PDF 文件處理與圖像提取
- AI 輔助元數據提取 (使用 Ollama)

### 📝 元數據管理
- 圖像標題和關鍵字
- 批量元數據編輯
- 數據庫持久化

### 🗂️ 圖庫瀏覽
- 查看所有索引的圖像和 PDF
- 按類型篩選
- 分頁瀏覽

### 🔐 用戶認證
- JWT 認證
- 管理員權限控制
- 用戶管理

## 安裝和運行

### 前置要求

- Python 3.10+
- Node.js 18+
- PostgreSQL
- (可選) Ollama (用於 AI 分析)

### 1. 設置後端

```bash
# 安裝 Python 依賴
pip install -r requirements-api.txt

# 設置數據庫
# 編輯 db_config.py 配置數據庫連接

# 初始化數據庫和創建管理員用戶
python auth.py

# 運行 FastAPI 服務器
uvicorn api.main:app --reload --port 8000
```

API 文檔將在 http://localhost:8000/docs 可用

### 2. 設置前端

```bash
cd frontend

# 安裝依賴
npm install

# 運行開發服務器
npm run dev
```

前端將在 http://localhost:3000 運行

### 3. (可選) 設置 Ollama

如果要使用 AI 分析 PDF:

```bash
# 安裝 Ollama (參考 OLLAMA_SETUP.md)

# 下載模型
ollama pull llama3.2:3b
```

## 項目結構

```
.
├── api/                      # FastAPI 後端
│   ├── main.py              # FastAPI 應用主文件
│   ├── routers/             # API 路由
│   │   ├── auth.py          # 認證相關
│   │   ├── search.py        # 搜索功能
│   │   ├── indexing.py      # 索引管理
│   │   ├── metadata.py      # 元數據管理
│   │   └── library.py       # 圖庫瀏覽
│   ├── models/              # Pydantic 模型
│   │   └── schemas.py       # 數據模型定義
│   └── services/            # 業務邏輯
│       └── clip_service.py  # CLIP 模型服務
│
├── frontend/                 # Next.js 前端
│   ├── src/
│   │   ├── app/             # Next.js App Router
│   │   │   ├── layout.tsx   # 根佈局
│   │   │   └── page.tsx     # 首頁
│   │   ├── components/      # React 組件
│   │   │   ├── LoginPage.tsx
│   │   │   ├── Navbar.tsx
│   │   │   ├── SearchPage.tsx
│   │   │   └── ImageGrid.tsx
│   │   └── lib/             # 工具庫
│   │       ├── api.ts       # API 客戶端
│   │       └── store.ts     # 狀態管理
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.js
│
├── images/                   # 圖像文件夾
├── catalog/                  # PDF 文件夾
├── metadata-files/           # 索引文件
├── config.json              # 配置文件
├── db_config.py             # 數據庫配置
├── db_helper.py             # 數據庫輔助函數
├── auth.py                  # 認證邏輯
├── pdf_utils.py             # PDF 處理工具
└── requirements-api.txt     # Python 依賴
```

## API 端點

### 認證
- `POST /api/auth/login` - 用戶登錄
- `GET /api/auth/me` - 獲取當前用戶信息
- `GET /api/auth/users` - 列出所有用戶 (管理員)

### 搜索
- `POST /api/search/text` - 文本搜索
- `POST /api/search/image` - 圖像搜索
- `GET /api/search/stats` - 搜索統計

### 索引
- `POST /api/indexing/sync` - 同步文件夾
- `POST /api/indexing/upload` - 上傳圖像
- `POST /api/indexing/upload-pdf` - 上傳 PDF
- `DELETE /api/indexing/remove` - 刪除圖像
- `GET /api/indexing/stats` - 索引統計

### 元數據
- `GET /api/metadata/all` - 獲取所有元數據
- `GET /api/metadata/{path}` - 獲取單個圖像元數據
- `POST /api/metadata/` - 保存元數據
- `PATCH /api/metadata/{path}` - 更新元數據
- `DELETE /api/metadata/{path}` - 刪除元數據

### 圖庫
- `GET /api/library/` - 獲取圖庫項目 (分頁)
- `GET /api/library/folders` - 列出可用文件夾

## 配置

編輯 `config.json`:

```json
{
  "folders": {
    "images": "images",
    "pdf_catalog": "catalog",
    "metadata": "metadata-files"
  },
  "model": {
    "name": "clip-vit-b-16",
    "architecture": "ViT-B-16",
    "pretrained": "openai",
    "embedding_dim": 512
  },
  "search": {
    "top_k": 8,
    "batch_size": 8
  },
  "pdf": {
    "max_keywords": 5,
    "jpeg_quality": 85,
    "ai_analysis": {
      "enabled": true,
      "model": "llama3.2:3b",
      "ollama_url": "http://localhost:11434"
    }
  }
}
```

## Docker 部署

```bash
# 構建和運行
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止
docker-compose down
```

## 默認用戶

- 用戶名: `admin`
- 密碼: `admin123`

⚠️ **重要**: 首次登錄後請立即修改密碼!

## 開發

### 後端開發

```bash
# 運行 FastAPI (熱重載)
uvicorn api.main:app --reload --port 8000

# 查看 API 文檔
# http://localhost:8000/docs
```

### 前端開發

```bash
cd frontend
npm run dev

# 前端將在 http://localhost:3000 運行
# 自動代理 API 請求到後端
```

## 技術棧

### 後端
- **FastAPI**: 現代 Python Web 框架
- **CLIP**: OpenAI 的圖像-文本模型
- **FAISS**: Facebook 的向量相似度搜索庫
- **PostgreSQL**: 關係數據庫
- **JWT**: JSON Web Tokens 認證

### 前端
- **Next.js 14**: React 框架 (App Router)
- **TypeScript**: 類型安全
- **Tailwind CSS**: 工具優先的 CSS 框架
- **Zustand**: 狀態管理
- **React Query**: 數據獲取和緩存
- **Axios**: HTTP 客戶端

## 從舊版 Streamlit 遷移

如果您之前使用 Streamlit 版本 (`app.py`):

1. 數據庫和索引文件保持兼容
2. 所有元數據自動遷移
3. PDF 和圖像文件無需移動
4. 配置文件 (`config.json`) 兼容

## 故障排除

### CLIP 模型加載失敗
確保已安裝 `open_clip_torch` 和 `torch`:
```bash
pip install open_clip_torch torch torchvision
```

### 數據庫連接錯誤
檢查 `db_config.py` 中的數據庫配置。

### 前端無法連接後端
確保後端運行在 `http://localhost:8000` 並檢查 CORS 設置。

### PDF 處理失敗
安裝 PDF 處理庫:
```bash
pip install PyPDF2 PyMuPDF pdf2image
```

## 許可證

MIT License

## 貢獻

歡迎提交 Pull Requests!

## 更新日誌

### v2.0.0 (2024-12-02)
- 🎉 完全重寫為 Next.js + FastAPI 架構
- ✨ 現代化 React UI
- 🚀 RESTful API 設計
- 🔐 JWT 認證
- 📱 響應式設計
- ⚡ 更好的性能
