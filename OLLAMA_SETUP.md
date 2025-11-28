# 本地 AI 分析設置指南 (Ollama + Llama 3)
# Local AI Analysis Setup Guide (Ollama + Llama 3)

## ✨ 優點 / Advantages

- ✅ **完全免費** - 無需付費 API
- ✅ **隱私保護** - 數據不離開本機
- ✅ **無限使用** - 沒有速率限制
- ✅ **離線工作** - 不需要網絡連接

## 📦 安裝步驟 / Installation

### 第 1 步：安裝 Ollama

**Windows:**
1. 下載：https://ollama.com/download/windows
2. 運行安裝程序
3. 安裝完成後，Ollama 會自動在後台運行

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 第 2 步：下載 AI 模型

打開命令提示符（cmd）並運行：

```cmd
# 推薦：Llama 3 (4.7GB，準確度高)
ollama pull llama3

# 或者：Mistral (4.1GB，速度快)
ollama pull mistral

# 或者：Llama 2 (3.8GB，較舊但穩定)
ollama pull llama2
```

**首次下載需要一些時間**（幾分鐘到十幾分鐘，取決於網速）

### 第 3 步：啟動 Ollama 服務

Ollama 通常會自動運行。如果沒有，手動啟動：

```cmd
ollama serve
```

保持這個窗口開著（或讓它在後台運行）

### 第 4 步：驗證安裝

```cmd
# 查看已安裝的模型
ollama list

# 測試模型
ollama run llama3 "Hello, how are you?"
```

### 第 5 步：安裝 Python 依賴

```cmd
pip install requests
```

## ⚙️ 配置 / Configuration

編輯 `config.json`：

```json
{
  "pdf": {
    "max_keywords": 5,
    "jpeg_quality": 85,
    "ai_analysis": {
      "enabled": true,              // 啟用 AI 分析
      "model": "llama3",            // 使用的模型：llama3, mistral, llama2
      "ollama_url": "http://localhost:11434",  // Ollama API 地址
      "fields": [
        "Project Name",
        "Location",
        "Client",
        "Role",
        "Date",
        "Description"
      ],
      "caption_template": "{project_name}"
    }
  }
}
```

## 🚀 使用方法 / Usage

1. **確保 Ollama 正在運行**
   ```cmd
   ollama serve
   ```

2. **運行測試**
   ```cmd
   python test_ai_analysis.py
   ```

3. **啟動應用**
   ```cmd
   streamlit run app.py
   ```

4. **上傳 PDF**
   - AI 會自動分析內容
   - 提取指定的字段
   - 生成智能標題

## 🎯 模型選擇 / Model Selection

| 模型 | 大小 | 速度 | 準確度 | 推薦用途 |
|------|------|------|--------|----------|
| **llama3** | 4.7GB | 中等 | 最高 | 推薦，最新最準確 |
| **mistral** | 4.1GB | 快 | 高 | 需要快速處理時 |
| **llama2** | 3.8GB | 中等 | 中 | 電腦配置較低時 |

切換模型只需在 `config.json` 中修改：
```json
"model": "mistral"  // 或 "llama3" 或 "llama2"
```

## 💻 系統需求 / System Requirements

**最低配置：**
- RAM: 8GB
- 硬盤空間: 10GB (用於模型)
- CPU: 現代多核處理器

**推薦配置：**
- RAM: 16GB+
- 硬盤: SSD
- CPU: 8 核心+
- GPU: 可選，但會顯著加快速度

## ⚡ 性能優化 / Performance Tips

### 使用 GPU 加速
如果你有 NVIDIA 顯卡：
```cmd
# Ollama 會自動使用 GPU（Windows/Linux）
ollama run llama3
```

### 調整模型大小
較小的模型變體：
```cmd
# Llama 3 8B (標準)
ollama pull llama3

# 或使用量化版本（更小、更快，但略低準確度）
ollama pull llama3:7b-q4_0
```

## 🔧 故障排除 / Troubleshooting

### 問題 1: "Cannot connect to Ollama"
**解決：**
```cmd
# 檢查 Ollama 是否運行
ollama serve

# 或重啟 Ollama 服務（Windows）
# 在任務管理器中找到 "Ollama" 並重啟
```

### 問題 2: 模型下載失敗
**解決：**
```cmd
# 清理並重新下載
ollama rm llama3
ollama pull llama3
```

### 問題 3: 分析速度很慢
**解決：**
- 使用較小的模型（mistral）
- 減少 PDF 文本長度
- 關閉其他占用資源的程序
- 考慮升級硬件（增加 RAM 或使用 GPU）

### 問題 4: JSON 解析錯誤
**原因：** 模型有時會返回不完整的 JSON
**解決：**
- 使用 llama3（最新，最穩定）
- 重試分析
- 檢查 PDF 文本質量

## 📊 性能對比 / Performance Comparison

| 方案 | 成本 | 速度 | 準確度 | 隱私 | 限制 |
|------|------|------|--------|------|------|
| **本地 Ollama** | 免費 | 中等 (10-30s) | 高 | 完全隱私 | 硬件要求 |
| OpenAI GPT-3.5 | $0.01-0.05/次 | 快 (2-5s) | 非常高 | 數據上傳 | 需付費 |
| OpenAI GPT-4 | $0.05-0.20/次 | 慢 (5-15s) | 最高 | 數據上傳 | 昂貴 |

## 🔄 切換回雲端 AI (可選)

如果你想使用 OpenAI GPT：
1. 安裝：`pip install openai`
2. 設置 API key：`set OPENAI_API_KEY=your-key`
3. 修改 `pdf_utils.py` 使用 OpenAI API

但對於個人使用和隱私保護，**本地方案更推薦**！

## 📝 測試命令

```cmd
# 測試 Ollama 安裝
ollama list

# 測試模型
ollama run llama3 "Extract project name from: Project ABC in Hong Kong"

# 測試 Python 集成
python test_ai_analysis.py

# 啟動應用
streamlit run app.py
```

## 🎓 進階使用

### 自定義提示詞
編輯 `pdf_utils.py` 中的 `prompt` 變量來優化提取效果。

### 使用其他模型
Ollama 支持多種模型：
```cmd
# 查看可用模型
ollama list

# 下載其他模型
ollama pull codellama    # 代碼專用
ollama pull neural-chat  # 對話優化
```

## 💡 小提示

1. **第一次使用會較慢** - 模型需要加載到內存
2. **保持 Ollama 運行** - 避免每次都重新啟動
3. **定期更新模型** - `ollama pull llama3` 會獲取最新版本
4. **監控資源使用** - 任務管理器查看 RAM/CPU 使用情況

需要幫助？查看 Ollama 文檔：https://ollama.com/docs
