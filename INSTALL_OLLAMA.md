# Ollama 安裝指南 (本地 AI)

## 📦 快速安裝 (3 步驟)

### 第 1 步：下載並安裝 Ollama

**Windows:**
1. 訪問：https://ollama.com/download/windows
2. 下載 `OllamaSetup.exe`
3. 運行安裝程序
4. 安裝完成後 Ollama 會自動啟動

### 第 2 步：下載 AI 模型

打開命令提示符（cmd）並運行：

```cmd
ollama pull llama3.2
```

**下載時間：** 約 2-5 分鐘（2GB 模型）

**其他模型選擇：**
```cmd
ollama pull llama3.2      # 推薦：最新，2GB，速度快
ollama pull llama3        # 4.7GB，更準確但較慢
ollama pull mistral       # 4.1GB，另一個選擇
```

### 第 3 步：測試安裝

```cmd
# 查看已安裝的模型
ollama list

# 測試模型
ollama run llama3.2 "Hello, extract the project name from this text: Working on Hong Kong Tower project"
```

## ✅ 完成！

現在可以使用 AI 分析功能了。

## 🚀 在應用中啟用

**1. 編輯 `config.json`:**
```json
{
  "pdf": {
    "ai_analysis": {
      "enabled": true,
      "model": "llama3.2"
    }
  }
}
```

**2. 啟動應用:**
```cmd
streamlit run app.py
```

**3. 上傳 PDF:**
- AI 會自動分析內容
- 即使沒有 "Project Name:" 這樣的標籤也能提取信息

## 💡 工作原理

Ollama 使用真正的 AI 語言模型來：
- **理解文本內容**
- **推斷項目名稱、地點、客戶等信息**
- **即使沒有明確標籤也能提取**

示例：
```
輸入 PDF 內容：
"We completed the renovation of the historic building 
in Central Hong Kong for ABC Development last year."

AI 自動提取：
✓ Project Name: Historic building renovation
✓ Location: Central Hong Kong
✓ Client: ABC Development
✓ Date: Last year
```

## ⚙️ Ollama 管理

### 啟動 Ollama
```cmd
ollama serve
```

### 停止 Ollama
在任務管理器中結束 "Ollama" 進程

### 查看模型
```cmd
ollama list
```

### 刪除模型
```cmd
ollama rm llama3.2
```

### 更新模型
```cmd
ollama pull llama3.2
```

## 📊 模型對比

| 模型 | 大小 | 速度 | 準確度 | 推薦場景 |
|------|------|------|--------|----------|
| **llama3.2** | 2GB | 快 (5-15s) | 高 | **推薦**，日常使用 |
| llama3 | 4.7GB | 中 (10-30s) | 非常高 | 需要最高準確度 |
| mistral | 4.1GB | 快 (8-20s) | 高 | 另一個選擇 |

## 💻 系統需求

**最低：**
- RAM: 8GB
- 硬盤: 5GB
- CPU: 4 核心

**推薦：**
- RAM: 16GB+
- 硬盤: SSD
- CPU: 8 核心+

## 🔧 故障排除

### 問題: "Ollama not running"
**解決：**
```cmd
ollama serve
```

### 問題: 模型下載失敗
**解決：**
```cmd
# 清理並重試
ollama rm llama3.2
ollama pull llama3.2
```

### 問題: 分析很慢
**解決：**
- 使用 llama3.2（最小最快）
- 確保沒有其他程序占用資源
- 考慮升級硬件

### 問題: "Invalid JSON response"
**原因：** 模型輸出格式不正確（偶爾發生）
**解決：**
- 重試分析
- 使用 llama3（更穩定）

## 🎯 使用建議

1. **首次使用很慢** - 模型需要加載到內存（約 10-20 秒）
2. **後續分析較快** - 模型已在內存中（5-15 秒）
3. **保持 Ollama 運行** - 避免每次重啟
4. **定期更新** - `ollama pull llama3.2` 獲取最新版本

## 📝 測試命令

```cmd
# 測試 Ollama
ollama run llama3.2 "Extract project info from: Building renovation in Hong Kong for ABC Corp"

# 測試 Python 集成
python test_ai_analysis.py

# 啟動應用
streamlit run app.py
```

## ✨ 優勢

✅ **真正的 AI 理解** - 不依賴固定格式
✅ **完全免費** - 無 API 費用
✅ **隱私保護** - 數據不上傳
✅ **離線工作** - 不需網絡
✅ **無限使用** - 沒有限制

開始使用：https://ollama.com
