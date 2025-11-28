# AI PDF Analysis Setup Guide

## 功能說明 / Features

此功能使用 OpenAI GPT 來分析 PDF 內容，自動提取結構化信息並生成智能標題。

This feature uses OpenAI GPT to analyze PDF content, automatically extract structured information, and generate smart captions.

## 安裝 / Installation

### 1. 安裝 OpenAI 套件
```bash
pip install openai
```

### 2. 獲取 OpenAI API Key
1. 訪問 https://platform.openai.com/api-keys
2. 創建新的 API key
3. 複製並保存 API key

### 3. 設置 API Key

#### 方法 A: 環境變數（本地開發）
**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Mac/Linux:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

#### 方法 B: Streamlit Secrets（推薦用於部署）
創建文件 `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

**注意：** 不要將此文件提交到 Git！

## 配置 / Configuration

編輯 `config.json` 來啟用和配置 AI 分析：

```json
{
  "pdf": {
    "max_keywords": 5,
    "jpeg_quality": 85,
    "ai_analysis": {
      "enabled": true,  // 設置為 true 啟用 AI 分析
      "fields": [
        "Project Name",
        "Location", 
        "Client",
        "Role",
        "Date",
        "Description"
      ],
      "caption_template": "{project_name}"  // 可自定義標題模板
    }
  }
}
```

### 自定義提取字段 / Custom Fields

你可以自定義要提取的字段：

```json
"fields": [
  "Project Name",
  "Location",
  "Client Name",
  "Your Role",
  "Project Duration",
  "Technologies Used",
  "Team Size",
  "Budget"
]
```

### 自定義標題模板 / Custom Caption Template

你可以組合多個字段來生成標題：

```json
// 只顯示項目名稱
"caption_template": "{project_name}"

// 顯示項目名稱和地點
"caption_template": "{project_name} - {location}"

// 顯示項目名稱、客戶和角色
"caption_template": "{project_name} | {client} | {role}"

// 更複雜的模板
"caption_template": "[{location}] {project_name} ({date})"
```

## 使用方法 / Usage

1. 啟用 AI 分析後，上傳 PDF 文件
2. 系統會自動：
   - 提取 PDF 中的文字
   - 使用 AI 分析內容
   - 提取指定的字段信息
   - 根據模板生成智能標題
3. 查看 AI 提取的信息（會顯示在展開面板中）
4. 標題會自動填入 caption 欄位（可手動修改）
5. 保存 metadata

## 費用說明 / Pricing

- 使用 OpenAI API 會產生費用
- GPT-3.5-turbo: 約 $0.0015 / 1K tokens (input) + $0.002 / 1K tokens (output)
- GPT-4: 約 $0.03 / 1K tokens (input) + $0.06 / 1K tokens (output)
- 一般 PDF 分析每次約 $0.01 - $0.05

## 切換模型 / Switch Model

如需更高準確度，可在 `pdf_utils.py` 中修改模型：

```python
response = openai.chat.completions.create(
    model="gpt-4",  # 改為 gpt-4（更準確但更貴）
    # 或 "gpt-3.5-turbo"（較便宜）
    messages=[...]
)
```

## 故障排除 / Troubleshooting

### 錯誤：API key not found
- 確認已設置 `OPENAI_API_KEY` 環境變數或 Streamlit secrets
- 重新啟動 Streamlit 應用

### 錯誤：Rate limit exceeded
- OpenAI API 有速率限制
- 等待幾秒後重試
- 或升級你的 OpenAI 帳戶

### AI 提取的信息不準確
- 確保 PDF 有足夠的文字內容
- 調整 `fields` 字段名稱使其更具體
- 考慮使用 GPT-4 代替 GPT-3.5

## 禁用 AI 分析 / Disable AI Analysis

如不需要 AI 分析，在 `config.json` 中設置：

```json
"ai_analysis": {
  "enabled": false
}
```

系統會回退到基本的關鍵詞提取方式。
