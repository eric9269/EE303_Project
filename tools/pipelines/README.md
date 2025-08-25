# 整合管道

此目錄包含用於自動化執行完整流程的整合工具。

## 檔案說明

### `train_and_test_pipeline.py`

- **功能**: 整合的訓練和測試管道
- **用途**: 自動執行完整的兩階段模型訓練和測試流程
- **特點**:
  - 自動化執行所有步驟
  - 完整的錯誤處理
  - 詳細的進度追蹤
  - 自動生成報告

## 使用方式

### 完整管道執行

```bash
python train_and_test_pipeline.py \
    --training_data_dir ../dataset_generators/training_data \
    --output_dir pipeline_results \
    --text_embedding_dim 512 \
    --image_embedding_dim 512 \
    --hidden_dim 256 \
    --output_dim 128 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --similarity_threshold 0.5 \
    --device auto
```

## 執行流程

### 步驟 1: 訓練 Triplet 模型

- 載入 Triplet 訓練資料
- 創建相似度模型
- 執行訓練循環
- 保存最佳模型和訓練歷史

### 步驟 2: 訓練分類模型

- 載入分類訓練資料
- 創建分類模型
- 執行訓練循環
- 保存最佳模型和訓練歷史

### 步驟 3: 測試模型

- 載入訓練好的模型
- 執行單個模型測試
- 執行兩階段組合測試
- 生成詳細的測試報告

### 步驟 4: 生成最終報告

- 整合所有結果
- 生成 Markdown 格式的報告
- 保存配置和性能指標

## 輸出結構

```
pipeline_results/
├── triplet_model/              # Triplet 模型訓練結果
│   ├── best_triplet_model.pth
│   ├── final_triplet_model.pth
│   ├── training_history.json
│   └── ...
├── classification_model/       # 分類模型訓練結果
│   ├── best_classification_model.pth
│   ├── final_classification_model.pth
│   ├── training_history.json
│   ├── classification_report.json
│   └── ...
├── test_results/              # 測試結果
│   ├── all_test_results.json
│   ├── similarity_test_results.json
│   ├── classification_test_results.json
│   ├── two_stage_test_results.json
│   └── ...
└── pipeline_report.md         # 最終報告
```

## 參數說明

### 輸入參數

- `--training_data_dir`: 訓練資料目錄路徑
- `--output_dir`: 輸出目錄路徑

### 模型參數

- `--text_embedding_dim`: 文字 embedding 維度
- `--image_embedding_dim`: 圖片 embedding 維度
- `--hidden_dim`: 隱藏層維度
- `--output_dim`: 輸出維度

### 訓練參數

- `--batch_size`: 批次大小
- `--epochs`: 訓練輪數
- `--learning_rate`: 學習率

### 測試參數

- `--similarity_threshold`: 相似度閾值
- `--device`: 設備 (auto/cpu/cuda)

## 報告內容

### 管道概述

- 執行時間和配置
- 各步驟的執行狀態
- 整體性能摘要

### 模型架構

- 相似度模型架構
- 分類模型架構
- 參數配置詳情

### 訓練結果

- 訓練損失曲線
- 驗證性能指標
- 最佳模型資訊

### 測試結果

- 單個模型性能
- 兩階段組合性能
- 詳細的評估指標

### 使用指南

- 模型載入方式
- 推理流程說明
- 參數調優建議

## 錯誤處理

### 自動錯誤處理

- 檔案不存在檢查
- 模型載入失敗處理
- 訓練中斷恢復

### 日誌記錄

- 詳細的執行日誌
- 錯誤訊息記錄
- 性能指標追蹤

### 檢查點機制

- 定期保存進度
- 可從中斷點恢復
- 結果驗證

## 自定義擴展

### 添加新的訓練步驟

```python
def custom_training_step():
    # 實現自定義訓練邏輯
    pass

# 在管道中添加
pipeline.add_step(custom_training_step)
```

### 修改報告格式

```python
def generate_custom_report():
    # 實現自定義報告生成
    pass
```

### 添加新的評估指標

```python
def custom_evaluation():
    # 實現自定義評估邏輯
    pass
```

## 性能優化

### 並行處理

- 支援多 GPU 訓練
- 批次處理優化
- 記憶體管理

### 監控和調優

- 即時性能監控
- 自動參數調優
- 資源使用優化

## 故障排除

### 常見問題

1. **記憶體不足**: 減少 batch_size 或使用梯度累積
2. **訓練時間過長**: 減少 epochs 或使用更小的模型
3. **結果不理想**: 調整學習率或模型架構

### 調優建議

- 使用驗證集進行超參數調優
- 嘗試不同的模型架構
- 調整資料預處理流程
