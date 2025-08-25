# EE303_Project

這是一個包含多種網路爬蟲和資料分析功能的專案，主要用於學習和研究目的。

## 專案結構

```
old_stuff/
├── README.md
├── src/
│   ├── scrapers/           # 爬蟲程式
│   │   ├── momo/          # MOMO購物網站爬蟲
│   │   │   ├── momo_basic_scraper.py      # 基礎MOMO爬蟲
│   │   │   ├── momo_advanced_scraper.py   # 進階MOMO爬蟲（支援多關鍵字）
│   │   │   └── momo_homework_template.py  # MOMO作業模板
│   │   ├── facebook/      # Facebook爬蟲
│   │   │   └── facebook_group_scraper.py  # Facebook社團成員爬蟲
│   │   └── stock/         # 股票資料爬蟲
│   │       ├── stock_analyzer.py          # 股票分析器（完整版）
│   │       └── stock_template.py          # 股票分析模板（練習版）
│   ├── data/              # 輸入資料檔案
│   │   ├── search_queries.xlsx            # 搜尋關鍵字資料
│   │   └── facebook_members.txt           # Facebook成員名單
│   ├── output/            # 輸出資料檔案
│   │   ├── stock_analysis_result.csv      # 股票分析結果
│   │   └── stock_raw_data.csv             # 股票原始資料
│   └── images/            # 圖片檔案
│       └── output_chart.jpg               # 輸出圖表
└── source/                # 原始檔案備份（空）
```

## 功能說明

### 1. MOMO 購物網站爬蟲 (`src/scrapers/momo/`)

- **momo_basic_scraper.py**: 基礎版本，爬取運動鞋相關商品資訊
- **momo_advanced_scraper.py**: 進階版本，支援多關鍵字搜尋和 Excel 檔案讀取
- **momo_homework_template.py**: 作業模板，包含填空練習

### 2. Facebook 社團爬蟲 (`src/scrapers/facebook/`)

- **facebook_group_scraper.py**: 爬取 Facebook 社團成員資訊，並與班級名冊進行比對

### 3. 股票資料分析 (`src/scrapers/stock/`)

- **stock_analyzer.py**: 完整的股票分析器，包含驗證碼處理和資料分析
- **stock_template.py**: 練習模板，包含填空練習

## 使用說明

### 環境需求

```bash
pip install pandas beautifulsoup4 requests selenium webdriver-manager ddddocr matplotlib numpy
```

### 執行範例

```bash
# 執行MOMO基礎爬蟲
python src/scrapers/momo/momo_basic_scraper.py

# 執行Facebook社團爬蟲
python src/scrapers/facebook/facebook_group_scraper.py

# 執行股票分析器
python src/scrapers/stock/stock_analyzer.py
```

## 注意事項

1. 請遵守網站的使用條款和 robots.txt 規範
2. 建議在執行爬蟲時加入適當的延遲時間
3. 股票分析器需要處理驗證碼，可能需要手動介入
4. Facebook 爬蟲需要提供帳號密碼，請注意安全性

## 授權

本專案僅供學習和研究使用，請勿用於商業用途。
