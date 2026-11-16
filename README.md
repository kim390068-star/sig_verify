# 簽名 AI Agent 評分與分類系統

上傳電子簽名圖片，呼叫真實 **Vision LLM**（OpenAI GPT-4o / xAI Grok Vision 等）進行：

- **分類 (Classification)**：single_signature / multiple_signatures / complex_flourish / simple_print / illegible_abstract / stamp_like
- **複雜度分數 (Complexity Score)**：0–100

沒有 API Key 時會自動 fallback 到 OpenCV 啟發式評分。

---

## 一次設定，永久使用（推薦）

### 1. 安裝依賴

```bash
cd signature_analyzer
pip install -r requirements.txt
```

### 2. 設定 API Key（只需要做一次）

```bash
# 複製範本
cp .env.example .env

# 用任何編輯器打開 .env，把 key 改成你的真實值
# 例如：
#   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
#   OPENAI_MODEL=gpt-4o
#   OPENAI_BASE_URL=https://api.openai.com/v1
```

**xAI 使用者範例：**
```
OPENAI_API_KEY=xai-xxxxxxxx
OPENAI_MODEL=grok-2-vision-1212
OPENAI_BASE_URL=https://api.x.ai/v1
```

### 3. 啟動

```bash
python3 app.py
```

之後每次只要執行 `python3 app.py` 就好，**不用再填 key**。

瀏覽器打開：http://127.0.0.1:5000

---

## 打分標準

### 分類
| 類型 | 說明 |
|------|------|
| `single_signature` | 清晰、單一、連貫的簽名筆跡 |
| `multiple_signatures` | 兩個或以上相對獨立的簽名區域 |
| `complex_flourish` | 華麗、大量花飾、迴圈、裝飾線條 |
| `simple_print` | 接近印刷體、筆畫極簡 |
| `illegible_abstract` | 難以辨認或抽象 |
| `stamp_like` | 接近印章風格 |

### 複雜度 0–100
- 0–24 極低 · 25–44 低 · 45–64 中等 · 65–84 高 · 85–100 極高

---

## API

`POST /api/analyze`

- form-data: `file` = 圖片
- 可選（會覆蓋 .env）：`api_key` / `base_url` / `model`

---

## 專案結構

```
signature_analyzer/
├── app.py              # 後端主程式
├── .env                # 你的 API Key（請自己填，不要提交到 git）
├── .env.example        # 範本
├── requirements.txt
├── templates/
│   └── index.html      # 前端
├── static/
└── uploads/            # 上傳的圖片暫存
```
