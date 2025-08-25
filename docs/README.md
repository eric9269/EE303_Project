# MySQL 檔案直接讀取工具

這個工具套件可以直接讀取 `xampp/mysql/data/product_matching` 目錄中的 MySQL 檔案，**無需啟動 MySQL 服務**。

## 主要功能

- **直接讀取檔案**: 無需啟動 MySQL 或 XAMPP 服務
- **跨平台支援**: Windows, macOS, Linux
- **高效能處理**: 快速處理大量資料
- **正確欄位結構**: 保持原始的 list 欄位結構
- **資料完整性**: 不破壞原始的資料組織方式

## 工具概覽

### 核心工具

1. **`simple_correct_parser.py`** - 正確的解析工具（推薦使用）
2. **`mysql_file_reader.py`** - 檔案掃描工具
3. **`view_correct_data.py`** - 資料查看工具

## 真實欄位結構

經過深入分析，我們發現原始表格的真正欄位結構如下：

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

## 重要發現

### 原始問題

- **錯誤認知**: 之前認為表格有 58-264 個欄位
- **實際情況**: 原始表格只有 **6 個欄位**，但包含 list 資料
- **解析問題**: 我們的提取工具將 list 欄位分解成了很多小欄位

### 解決方案

- **正確解析**: 重新組織資料，保持原始的 list 結構
- **欄位恢復**: 將分解的資料重新組合為正確的 6 個欄位
- **資料完整性**: 保持原始的資料組織方式

## 檔案結構

```
tools/
├── simple_correct_parser.py    # 正確的解析工具
├── mysql_file_reader.py        # 檔案掃描工具
├── view_correct_data.py        # 資料查看工具
└── requirements.txt            # Python 依賴

data/
└── correct_structure_data/     # 正確結構的資料

docs/
└── README.md                   # 詳細技術文件
```

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 執行正確解析

```bash
cd tools
python simple_correct_parser.py
```

### 3. 查看結果

解析結果將保存在 `data/correct_structure_data/` 目錄中，包含：

- `leaf_v1_correct.csv` - LEAF v1 表格
- `root_v1_correct.csv` - ROOT v1 表格
- `leaf_it_shopee_correct.csv` - LEAF IT Shopee 表格
- `root_it_shopee_correct.csv` - ROOT IT Shopee 表格

## 資料統計

### LEAF 表格統計

- **平均圖片 URL 數**: 17.8 個
- **平均產品 ID 數**: 5.0 個
- **平均頁面 URL 數**: 3.5 個

### ROOT 表格統計

- **平均圖片 URL 數**: 31.1 個
- **平均產品 ID 數**: 6.4 個
- **平均頁面 URL 數**: 5.8 個

## 使用場景

1. **產品匹配系統**: 分析產品圖片和頁面連結
2. **資料分析**: 統計產品資訊和商店資料
3. **系統遷移**: 從 MySQL 檔案直接提取資料
4. **備份恢復**: 無需啟動服務即可讀取資料

## 注意事項

1. **檔案路徑**: 確保 `xampp/mysql/data/product_matching` 目錄存在
2. **檔案權限**: 確保有讀取 MySQL 檔案的權限
3. **資料完整性**: 解析結果保持原始的 list 結構
4. **效能考量**: 大檔案可能需要較長處理時間

## 技術細節

### 解析方法

- **二進制讀取**: 直接讀取 `.ibd` 檔案
- **文字提取**: 提取可讀的文字內容
- **正則表達式**: 識別 URL、產品 ID、價格等
- **資料重組**: 將分解的資料重新組織為正確結構

### 支援的資料類型

- **圖片 URL**: `https://gcs.rimg.com.tw/...`
- **產品頁面**: `https://goods.ruten.com.tw/item/show?...`
- **產品 ID**: 10 位以上數字
- **價格範圍**: `數字 - 數字` 格式
- **商店名稱**: 包含 `ruten`、`shopee`、`momo` 的識別碼

## 更新日誌

### v2.0 (最新)

- 修正欄位結構認知
- 實現正確的 list 欄位解析
- 移除舊的錯誤解析工具
- 更新文件說明

### v1.0 (舊版)

- 錯誤的欄位分解
- 破壞原始資料結構
- 產生過多的欄位

## 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個工具。

## 授權

本專案採用 MIT 授權條款。
