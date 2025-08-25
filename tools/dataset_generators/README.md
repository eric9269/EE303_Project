# 資料集生成器

此目錄包含用於生成機器學習訓練資料集的工具。

## 檔案說明

### `bm25_sampler.py`

- **功能**: BM25 樣本選擇器
- **用途**: 使用 TF-IDF 和餘弦相似度選擇正負樣本
- **特點**:
  - 基於 BM25 算法的相似度計算
  - 可配置的 k 值（負樣本數量）
  - 支援正負樣本平衡

### `classification_dataset_generator.py`

- **功能**: 分類訓練集生成器
- **用途**: 生成用於二分類任務的訓練資料
- **輸出格式**: [label, product_name_a, product_name_b, text_embedding_a, text_embedding_b, image_embedding_a, image_embedding_b]

### `triplet_dataset_generator.py`

- **功能**: Triplet 訓練集生成器
- **用途**: 生成用於相似度學習的 triplet 資料
- **輸出格式**: [anchor_text, positive_text, negative_text, anchor_text_embedding, positive_text_embedding, negative_text_embedding, anchor_image_embedding, positive_image_embedding, negative_image_embedding]

### `generate_training_datasets.py`

- **功能**: 整合訓練集生成腳本
- **用途**: 自動執行完整的資料集生成流程
- **特點**: 包含報告生成和進度追蹤

## 使用方式

### 單獨使用各生成器

```bash
# 生成 BM25 樣本
python bm25_sampler.py \
    --leaf_file ../data/correct_structure_data/leaf_v1_correct.csv \
    --root_file ../data/correct_structure_data/root_v1_correct.csv \
    --output samples.json \
    --k 5 \
    --sample_size 1000

# 生成分類訓練集
python classification_dataset_generator.py \
    --samples_file samples.json \
    --leaf_file ../data/correct_structure_data/leaf_v1_correct.csv \
    --root_file ../data/correct_structure_data/root_v1_correct.csv \
    --output classification_dataset.csv \
    --embedding_dim 512 \
    --use_clip

# 生成 Triplet 訓練集
python triplet_dataset_generator.py \
    --samples_file samples.json \
    --leaf_file ../data/correct_structure_data/leaf_v1_correct.csv \
    --root_file ../data/correct_structure_data/root_v1_correct.csv \
    --output triplet_dataset.csv \
    --embedding_dim 512 \
    --triplet_per_anchor 3 \
    --max_triplets 10000 \
    --use_clip
```

### 使用整合腳本

```bash
# 生成所有訓練資料集
python generate_training_datasets.py \
    --leaf_file ../data/correct_structure_data/leaf_v1_correct.csv \
    --root_file ../data/correct_structure_data/root_v1_correct.csv \
    --output_dir training_data \
    --k 5 \
    --sample_size 1000 \
    --embedding_dim 512 \
    --triplet_per_anchor 3 \
    --max_triplets 10000 \
    --use_clip
```

## 資料格式

### BM25 樣本格式 (JSON)

```json
{
  "positive_samples": [[leaf_idx, root_idx], ...],
  "negative_samples": [[leaf_idx, root_idx], ...],
  "config": {
    "k": 5,
    "sample_size": 1000,
    "similarity_scores": {...}
  }
}
```

### 分類訓練集格式 (CSV)

| 欄位              | 說明                      |
| ----------------- | ------------------------- |
| label             | 標籤 (0: 不匹配, 1: 匹配) |
| product_name_a    | 商品 A 名稱               |
| product_name_b    | 商品 B 名稱               |
| text_embedding_a  | 商品 A 文字 embedding     |
| text_embedding_b  | 商品 B 文字 embedding     |
| image_embedding_a | 商品 A 圖片 embedding     |
| image_embedding_b | 商品 B 圖片 embedding     |

### Triplet 訓練集格式 (CSV)

| 欄位                     | 說明                    |
| ------------------------ | ----------------------- |
| anchor_text              | Anchor 商品名稱         |
| positive_text            | Positive 商品名稱       |
| negative_text            | Negative 商品名稱       |
| anchor_text_embedding    | Anchor 文字 embedding   |
| positive_text_embedding  | Positive 文字 embedding |
| negative_text_embedding  | Negative 文字 embedding |
| anchor_image_embedding   | Anchor 圖片 embedding   |
| positive_image_embedding | Positive 圖片 embedding |
| negative_image_embedding | Negative 圖片 embedding |

## 參數說明

### BM25 參數

- `--k`: 每個正樣本對應的負樣本數量
- `--sample_size`: 總樣本數量限制

### 分類訓練集參數

- `--embedding_dim`: Embedding 維度
- `--max_text_length`: 文字最大長度
- `--use_clip`: 是否使用 CLIP 模型

### Triplet 訓練集參數

- `--triplet_per_anchor`: 每個 anchor 生成的 triplet 數量
- `--max_triplets`: 最大 triplet 數量

## 注意事項

- 確保輸入的 LEAF 和 ROOT 檔案存在且格式正確
- 使用 CLIP 時需要網路連接
- 大量資料處理時注意記憶體使用
- 建議根據實際需求調整樣本數量和比例
