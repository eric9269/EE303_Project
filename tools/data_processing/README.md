# 資料處理工具

此目錄包含用於處理和解析 MySQL 資料檔案的工具。

## 檔案說明

### `simple_correct_parser.py`

- **功能**: 主要解析工具，將 MySQL 資料重新組織為正確的 6 欄結構
- **用途**: 處理 `.ibd` 檔案，保持原始的 list 資料結構
- **輸出**: 正確結構的 CSV 檔案

### `view_correct_data.py`

- **功能**: 資料查看工具
- **用途**: 檢視解析後的資料結構和內容
- **特點**: 支援 list 欄位的正確顯示

### `mysql_file_reader.py`

- **功能**: 檔案掃描工具
- **用途**: 掃描和列出 MySQL 資料檔案
- **特點**: 無需啟動 MySQL 服務即可讀取檔案

## 使用方式

```bash
# 解析資料
python simple_correct_parser.py

# 檢視資料
python view_correct_data.py

# 掃描檔案
python mysql_file_reader.py
```

## 注意事項

- 所有工具都設計為無需啟動 MySQL 服務
- 保持原始資料的完整性，不進行不必要的清理
- 支援 list 類型的欄位資料
