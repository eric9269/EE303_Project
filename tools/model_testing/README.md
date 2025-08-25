# 模型測試

此目錄包含用於測試和評估模型的工具。

## 檔案說明

### `test_models.py`

- **功能**: 模型測試程式
- **用途**: 測試單個模型和兩階段組合模型
- **支援的測試**:
  - 相似度模型測試
  - 分類模型測試
  - 兩階段組合模型測試

## 使用方式

### 測試單個模型

```bash
# 測試相似度模型
python test_models.py \
    --test_data_path ../dataset_generators/triplet_dataset.csv \
    --similarity_model_path ../model_training/triplet_model/best_triplet_model.pth \
    --output_dir test_results \
    --batch_size 32 \
    --device auto

# 測試分類模型
python test_models.py \
    --test_data_path ../dataset_generators/classification_dataset.csv \
    --classification_model_path ../model_training/classification_model/best_classification_model.pth \
    --output_dir test_results \
    --batch_size 32 \
    --device auto
```

### 測試兩階段組合模型

```bash
python test_models.py \
    --test_data_path ../dataset_generators/classification_dataset.csv \
    --similarity_model_path ../model_training/triplet_model/best_triplet_model.pth \
    --classification_model_path ../model_training/classification_model/best_classification_model.pth \
    --output_dir test_results \
    --similarity_threshold 0.5 \
    --batch_size 32 \
    --device auto
```

## 測試結果

### 相似度模型測試結果

```json
{
  "avg_positive_similarity": 0.8234,
  "avg_negative_similarity": 0.2341,
  "similarity_gap": 0.5893,
  "accuracy": 0.9456,
  "total_samples": 1000,
  "correct_predictions": 945
}
```

### 分類模型測試結果

```json
{
  "accuracy": 0.9234,
  "precision": 0.9156,
  "recall": 0.9289,
  "f1": 0.9222,
  "classification_report": {
    "Negative": {
      "precision": 0.9312,
      "recall": 0.9187,
      "f1-score": 0.9249
    },
    "Positive": {
      "precision": 0.9,
      "recall": 0.9391,
      "f1-score": 0.9192
    }
  },
  "total_samples": 1000
}
```

### 兩階段組合模型測試結果

```json
{
  "accuracy": 0.9345,
  "precision": 0.9289,
  "recall": 0.9378,
  "f1": 0.9333,
  "avg_similarity": 0.7123,
  "stage1_pass_rate": 0.4567,
  "similarity_threshold": 0.5,
  "total_samples": 1000,
  "stage1_passed": 456
}
```

## 測試流程

### 相似度模型測試流程

1. **載入模型**: 載入訓練好的相似度模型
2. **載入資料**: 載入 Triplet 測試資料
3. **計算相似度**: 計算 anchor 與 positive/negative 的相似度
4. **統計分析**: 計算平均相似度、相似度差距等指標
5. **準確率計算**: 基於相似度比較計算準確率

### 分類模型測試流程

1. **載入模型**: 載入訓練好的分類模型
2. **載入資料**: 載入分類測試資料
3. **預測**: 對測試資料進行預測
4. **指標計算**: 計算準確率、精確率、召回率、F1 分數
5. **分類報告**: 生成詳細的分類報告

### 兩階段組合模型測試流程

1. **載入模型**: 載入相似度模型和分類模型
2. **第一階段**: 計算商品對的相似度，根據閾值篩選
3. **第二階段**: 對通過第一階段的候選對進行分類
4. **組合預測**: 結合兩個階段的結果
5. **性能評估**: 計算最終的性能指標

## 評估指標

### 相似度模型指標

- **平均正樣本相似度**: 正樣本對的平均相似度
- **平均負樣本相似度**: 負樣本對的平均相似度
- **相似度差距**: 正負樣本相似度的差距
- **準確率**: 正樣本相似度 > 負樣本相似度的比例

### 分類模型指標

- **準確率 (Accuracy)**: 正確預測的比例
- **精確率 (Precision)**: 預測為正樣本中實際為正樣本的比例
- **召回率 (Recall)**: 實際正樣本中被正確預測的比例
- **F1 分數**: 精確率和召回率的調和平均

### 兩階段模型指標

- **平均相似度**: 所有商品對的平均相似度
- **第一階段通過率**: 通過相似度閾值的比例
- **最終性能指標**: 組合後的準確率、精確率、召回率、F1 分數

## 參數說明

### 測試參數

- `--test_data_path`: 測試資料路徑
- `--similarity_model_path`: 相似度模型路徑
- `--classification_model_path`: 分類模型路徑
- `--output_dir`: 測試結果輸出目錄
- `--similarity_threshold`: 相似度閾值 (兩階段模型)
- `--batch_size`: 批次大小
- `--device`: 設備 (auto/cpu/cuda)

## 輸出檔案

### 測試結果檔案

- `all_test_results.json`: 所有測試結果的總覽
- `similarity_test_results.json`: 相似度模型測試結果
- `classification_test_results.json`: 分類模型測試結果
- `two_stage_test_results.json`: 兩階段組合模型測試結果

## 性能分析

### 相似度分析

- 正負樣本相似度分布
- 相似度閾值對性能的影響
- 模型學習效果評估

### 分類性能分析

- 混淆矩陣分析
- 各類別的詳細性能
- 錯誤案例分析

### 兩階段性能分析

- 各階段的貢獻度
- 閾值調優建議
- 整體性能評估

## 故障排除

### 常見問題

1. **模型載入失敗**: 檢查模型路徑和格式
2. **記憶體不足**: 減少 batch_size
3. **資料格式錯誤**: 檢查測試資料格式

### 調優建議

- 調整相似度閾值以平衡精確率和召回率
- 分析錯誤案例以改進模型
- 使用不同的評估指標進行綜合分析
