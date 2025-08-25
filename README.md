# EE303 Project - MySQL Data Parser & ML Training Dataset Generator

這是一個能「直接讀取 MySQL 檔案」並「自動生成機器學習訓練資料集」的工具，專為產品匹配（Product Matching）系統設計。簡單來說，它就像是一個能幫你快速打開 MySQL 檔案的鑰匙，再幫你整理好資料，變成能直接拿去訓練 AI 模型的乾淨資料集。

## 專案概述

這個 Project 的核心目標是：可以不用啟動 MySQL 伺服器，就能直接處理我之前讓學弟妹標注的資料，並生成適合分類或相似度學習的訓練資料集。

- 直接讀取檔案：不需要啟動 MySQL 或 XAMPP，就能直接從 .ibd 檔案裡取資料。
- 跨平台支援：Windows / macOS / Linux 都能跑。
- 高效能處理：面對大資料集依然能快速解析。
- 保持資料完整性：原始欄位結構不會被破壞。
- 機器學習友善：自動生成可用於 分類 和 相似度學習 的訓練資料集。
- CLIP 模型整合：可直接將圖片轉換成 embedding，支援最新的 EVA-CLIP。

## 為什麼需要這個工具？

由於舊的資料遺失了，為了處理產品匹配的原始 SQL 資料，並克服下面兩個問題：
1. MySQL 資料難以直接存取：傳統方式需要啟動 MySQL，再透過 SQL 指令查詢，過程繁瑣又耗時。
2.訓練資料集準備困難：要人工清理、標記、轉換資料，才能拿來訓練模型。


## 專案結構

```
EE303_Project/
├── README.md                 # 專案主要說明文件
├── requirements.txt          # Python 依賴套件
├── tools/                    # 核心工具
│   ├── data_processing/           # 資料處理工具
│   │   ├── simple_correct_parser.py
│   │   ├── view_correct_data.py
│   │   └── mysql_file_reader.py
│   ├── embedding_processors/      # Embedding 處理器
│   │   ├── clip_image_processor.py
│   │   └── text_embedding_processor.py
│   ├── dataset_generators/        # 資料集生成器
│   │   ├── bm25_sampler.py
│   │   ├── classification_dataset_generator.py
│   │   ├── triplet_dataset_generator.py
│   │   └── generate_training_datasets.py
│   ├── model_training/           # 模型訓練
│   │   ├── train_triplet_model.py
│   │   └── train_classification_model.py
│   ├── model_testing/            # 模型測試
│   │   └── test_models.py
│   ├── pipelines/                # 整合管道
│   │   └── train_and_test_pipeline.py
│   └── README.md                 # 工具說明文件
├── data/                     # 解析後的資料
│   └── correct_structure_data/     # 正確結構的資料
├── docs/                     # 文件
│   ├── README.md             # 詳細技術文件
│   └── training_dataset_guide.md   # 訓練資料集使用指南
├── src/                      # 其他專案模組
│   └── class_stuff/              # 課程相關工具
└── xampp/                    # XAMPP 資料目錄
    └── mysql/data/product_matching/  # MySQL 原始資料
```

## 快速開始

### 1. 安裝依賴

```bash
# 使用 Makefile 安裝
make install

# 或手動安裝
pip install -r requirements.txt
```

### 2. 處理 MySQL 資料

```bash
# 使用 Makefile 處理資料
make data-processing

# 或手動執行
cd tools/data_processing
python simple_correct_parser.py
```

### 3. 檢視處理後的資料

```bash
# 使用 Makefile 檢視資料
make view-data

# 或手動執行
cd tools/data_processing
python view_correct_data.py
```

### 4. 生成訓練資料集（可選）

```bash
# 使用 Makefile 生成所有訓練資料集
make dataset-generation

# 或者單獨生成
make bm25-samples
make classification-dataset
make triplet-dataset
```

### 5. 訓練和測試兩階段模型（可選）

```bash
# 使用 Makefile 執行完整管道
make pipeline

# 或者單獨執行
make model-training
make model-testing
```

### 6. 使用 Makefile 管理工具

```bash
# 查看所有可用命令
make help

# 驗證所有工具
make validate

# 運行基本測試
make test

# 清理暫存檔案
make clean
```

## 資料結構

### LEAF 表格 (葉節點 - 產品詳細資料)

| 欄位名稱      | 內容類型          | 說明             | 平均數量 |
| ------------- | ----------------- | ---------------- | -------- |
| `product_ids` | 產品識別碼列表    | 產品 ID 列表     | 5.0 個   |
| `image_urls`  | 圖片 URL 列表     | 產品圖片連結     | 17.8 個  |
| `page_urls`   | 產品頁面 URL 列表 | 產品頁面連結     | 3.5 個   |
| `prices`      | 價格資訊列表      | 價格範圍         | 0.0 個   |
| `shops`       | 商店資訊列表      | 商店名稱         | 變動     |
| `link`        | 對應的 ROOT ID    | 關聯到 ROOT 表格 | 變動     |

### ROOT 表格 (根節點 - 產品分類資料)

| 欄位名稱      | 內容類型          | 說明         | 平均數量 |
| ------------- | ----------------- | ------------ | -------- |
| `product_ids` | 產品識別碼列表    | 分類產品 ID  | 6.4 個   |
| `image_urls`  | 圖片 URL 列表     | 分類圖片連結 | 31.1 個  |
| `page_urls`   | 產品頁面 URL 列表 | 分類產品連結 | 5.8 個   |
| `prices`      | 價格資訊列表      | 分類價格範圍 | 0.0 個   |
| `shops`       | 商店資訊列表      | 商店名稱     | 變動     |

## 核心功能

### 1. MySQL 檔案解析

- **直接讀取**: 無需啟動 MySQL 服務即可讀取 `.ibd` 檔案
- **欄位重組**: 將分解的欄位重新組織為正確的 6 欄結構
- **資料完整性**: 保持原始的 list 資料結構

### 2. 訓練資料集生成

- **BM25 樣本選擇**: 使用 TF-IDF 和餘弦相似度選擇正負樣本
- **分類訓練集**: 生成用於二分類任務的訓練資料
- **Triplet 訓練集**: 生成用於相似度學習的 triplet 資料
- **CLIP 整合**: 支援 EVA-CLIP 和標準 CLIP 模型生成高品質圖片 embedding

### 3. 兩階段模型訓練

- **第一階段**: Triplet 相似度模型訓練
- **第二階段**: Classification 分類模型訓練
- **模型架構**: 可配置的神經網路架構
- **自動化管道**: 完整的訓練和測試流程

### 4. 機器學習支援

- **文字 Embedding**: 支援 BERT、TF-IDF 和自定義模型
- **圖片 Embedding**: 支援 EVA-CLIP 和標準 CLIP 模型（可選隨機向量）
- **資料正規化**: 所有 embedding 都已正規化
- **批次處理**: 支援大規模資料的批次處理

## 使用場景

1. **產品匹配系統**: 分析產品圖片和頁面連結
2. **資料分析**: 統計產品資訊和商店資料
3. **系統遷移**: 從 MySQL 檔案直接提取資料
4. **備份恢復**: 無需啟動服務即可讀取資料
5. **機器學習訓練**: 生成分類和相似度學習的訓練資料集
6. **產品推薦**: 基於圖片和文字相似度的推薦系統

## 技術特點

- **二進制讀取**: 直接讀取 `.ibd` 檔案
- **文字提取**: 提取可讀的文字內容
- **正則表達式**: 識別 URL、產品 ID、價格等
- **資料重組**: 將分解的資料重新組織為正確結構
- **CLIP 整合**: 使用最先進的視覺-語言模型
- **批次處理**: 高效處理大規模資料集

## 支援的資料類型

- **圖片 URL**: `https://gcs.rimg.com.tw/...`
- **產品頁面**: `https://goods.ruten.com.tw/item/show?...`
- **產品 ID**: 10 位以上數字
- **價格範圍**: `數字 - 數字` 格式
- **商店名稱**: 包含 `ruten`、`shopee`、`momo` 的識別碼

## 效能優化

- **記憶體管理**: 分批處理大檔案以減少記憶體使用
- **並行處理**: 支援多進程處理以提高效能
- **快取機制**: 避免重複計算 embedding
- **網路優化**: 圖片下載時添加延遲避免過度請求

## 注意事項

1. **檔案路徑**: 確保 `xampp/mysql/data/product_matching` 目錄存在
2. **檔案權限**: 確保有讀取 MySQL 檔案的權限
3. **資料完整性**: 解析結果保持原始的 list 結構
4. **效能考量**: 大檔案可能需要較長處理時間
5. **網路連接**: 使用 CLIP 時需要網路連接下載圖片
6. **GPU 支援**: CLIP 模型支援 GPU 加速（如果可用）

## 系統需求

- **Python**: 3.8+
- **記憶體**: 建議 8GB+（處理大資料集時）
- **儲存空間**: 根據資料集大小而定
- **網路**: 使用 CLIP 功能時需要穩定的網路連接
- **GPU**: 可選，用於加速 CLIP 模型推理

## 故障排除

### 常見問題

1. **記憶體不足**: 減少 `sample_size` 或 `embedding_dim` 參數
2. **網路錯誤**: 檢查網路連接，或使用 `--no_clip` 選項
3. **檔案不存在**: 檢查輸入檔案路徑是否正確
4. **權限錯誤**: 確保有寫入輸出目錄的權限

### 效能調優

1. **批次大小**: 調整批次處理大小以平衡記憶體和效能
2. **並行度**: 根據 CPU 核心數調整並行處理數量
3. **快取策略**: 對於重複使用的 embedding 啟用快取

### 貢獻指南

1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 授權

本專案採用 MIT 授權條款。

## 詳細文件

- **技術細節**: [docs/README.md](docs/README.md)
- **訓練資料集使用指南**: [docs/training_dataset_guide.md](docs/training_dataset_guide.md)
