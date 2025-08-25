# 模型訓練

此目錄包含用於訓練兩階段商品匹配模型的工具。

## 檔案說明

### `train_triplet_model.py`

- **功能**: Triplet 模型訓練程式
- **用途**: 訓練第一階段的相似度模型
- **模型架構**:
  - 文字和圖片編碼器
  - 特徵融合層
  - Triplet Loss
- **輸出**: 128 維相似度 embedding

### `train_classification_model.py`

- **功能**: 分類模型訓練程式
- **用途**: 訓練第二階段的分類模型
- **模型架構**:
  - 文字和圖片編碼器
  - 特徵差異計算
  - 分類層
- **輸出**: 二分類結果 (0: 不匹配, 1: 匹配)

## 使用方式

### 訓練 Triplet 模型

```bash
python train_triplet_model.py \
    --data_path ../dataset_generators/triplet_dataset.csv \
    --output_dir triplet_model \
    --text_embedding_dim 512 \
    --image_embedding_dim 512 \
    --hidden_dim 256 \
    --output_dim 128 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --device auto
```

### 訓練分類模型

```bash
python train_classification_model.py \
    --data_path ../dataset_generators/classification_dataset.csv \
    --output_dir classification_model \
    --text_embedding_dim 512 \
    --image_embedding_dim 512 \
    --hidden_dim 256 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --device auto
```

## 模型架構詳解

### SimilarityModel (相似度模型)

```python
class SimilarityModel(nn.Module):
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512,
                 hidden_dim=256, output_dim=128):
        super().__init__()

        # 文字處理層
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.L2Normalization()
        )

        # 圖片處理層
        self.image_encoder = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.L2Normalization()
        )

        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.L2Normalization()
        )
```

### ClassificationModel (分類模型)

```python
class ClassificationModel(nn.Module):
    def __init__(self, text_embedding_dim=512, image_embedding_dim=512,
                 hidden_dim=256, num_classes=2):
        super().__init__()

        # 文字處理層
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 圖片處理層
        self.image_encoder = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 特徵融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
```

## 訓練參數

### 通用參數

- `--data_path`: 訓練資料路徑
- `--output_dir`: 模型輸出目錄
- `--text_embedding_dim`: 文字 embedding 維度
- `--image_embedding_dim`: 圖片 embedding 維度
- `--hidden_dim`: 隱藏層維度
- `--batch_size`: 批次大小
- `--epochs`: 訓練輪數
- `--learning_rate`: 學習率
- `--train_split`: 訓練集比例
- `--device`: 設備 (auto/cpu/cuda)

### Triplet 模型特有參數

- `--output_dim`: 輸出維度

## 輸出檔案

### 訓練過程檔案

- `best_*.pth`: 最佳模型權重
- `final_*.pth`: 最終模型權重
- `*_epoch_*.pth`: 檢查點檔案
- `training_history.json`: 訓練歷史記錄

### 分類模型額外檔案

- `classification_report.json`: 分類報告

## 訓練監控

### 損失曲線

訓練過程中會記錄並保存：

- 訓練損失
- 驗證損失
- 學習率變化

### 早停機制

- 自動保存最佳模型
- 基於驗證損失的早停
- 定期檢查點保存

## 性能優化

### GPU 加速

```bash
--device cuda
```

### 混合精度訓練

```python
# 在訓練代碼中已實現
scaler = torch.cuda.amp.GradScaler()
```

### 記憶體優化

- 梯度累積
- 動態批次大小
- 記憶體清理

## 故障排除

### 常見問題

1. **記憶體不足**: 減少 batch_size 或 hidden_dim
2. **訓練不收斂**: 調整 learning_rate 或增加 epochs
3. **過擬合**: 增加 Dropout 率或減少模型複雜度

### 調優建議

- 使用驗證集監控訓練
- 嘗試不同的學習率調度器
- 調整模型架構參數
