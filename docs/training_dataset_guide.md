# 訓練資料集生成工具使用指南

本指南說明如何使用我們提供的工具生成用於機器學習的訓練資料集。

## 工具概述

我們提供了四個主要的工具來生成不同類型的訓練資料：

1. **BM25 樣本選擇器** (`bm25_sampler.py`)
2. **分類訓練集生成器** (`classification_dataset_generator.py`)
3. **Triplet 訓練集生成器** (`triplet_dataset_generator.py`)
4. **CLIP 圖片處理器** (`clip_image_processor.py`)
5. **整合腳本** (`generate_training_datasets.py`)

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 使用整合腳本（推薦）

```bash
cd tools
python generate_training_datasets.py \
    --leaf_file ../data/correct_structure_data/leaf_v1_correct.csv \
    --root_file ../data/correct_structure_data/root_v1_correct.csv \
    --output_dir training_data \
    --k 5 \
    --sample_size 1000 \
    --embedding_dim 512 \
    --use_clip \
    --max_triplets 10000
```

## 詳細參數說明

### BM25 樣本選擇器

```bash
python bm25_sampler.py \
    --leaf_file <LEAF_CSV_PATH> \
    --root_file <ROOT_CSV_PATH> \
    --k <NEGATIVE_SAMPLES_PER_LEAF> \
    --sample_size <TOTAL_SAMPLE_SIZE> \
    --output <OUTPUT_JSON_FILE>
```

**參數說明：**

- `--leaf_file`: LEAF 表格的 CSV 檔案路徑
- `--root_file`: ROOT 表格的 CSV 檔案路徑
- `--k`: 每個 leaf 選擇的負樣本數量（預設：5）
- `--sample_size`: 總樣本數量限制（預設：1000）
- `--output`: 輸出 JSON 檔案路徑（預設：samples.json）

### 分類訓練集生成器

```bash
python classification_dataset_generator.py \
    --samples_file <SAMPLES_JSON_FILE> \
    --leaf_file <LEAF_CSV_PATH> \
    --root_file <ROOT_CSV_PATH> \
    --output <OUTPUT_CSV_FILE> \
    --embedding_dim <EMBEDDING_DIMENSION> \
    --max_text_length <MAX_TEXT_LENGTH>
```

**參數說明：**

- `--samples_file`: BM25 樣本檔案路徑
- `--leaf_file`: LEAF 表格的 CSV 檔案路徑
- `--root_file`: ROOT 表格的 CSV 檔案路徑
- `--output`: 輸出 CSV 檔案路徑（預設：classification_dataset.csv）
- `--embedding_dim`: 文字 Embedding 維度（預設：512）
- `--max_text_length`: 文字最大長度（預設：100）
- `--use_clip`: 使用 CLIP 模型（預設：True）
- `--no_clip`: 不使用 CLIP 模型，使用隨機向量

### Triplet 訓練集生成器

```bash
python triplet_dataset_generator.py \
    --samples_file <SAMPLES_JSON_FILE> \
    --leaf_file <LEAF_CSV_PATH> \
    --root_file <ROOT_CSV_PATH> \
    --output <OUTPUT_CSV_FILE> \
    --embedding_dim <EMBEDDING_DIMENSION> \
    --max_text_length <MAX_TEXT_LENGTH> \
    --triplet_per_anchor <TRIPLETS_PER_ANCHOR> \
    --max_triplets <MAX_TRIPLETS>
```

**參數說明：**

- `--samples_file`: BM25 樣本檔案路徑
- `--leaf_file`: LEAF 表格的 CSV 檔案路徑
- `--root_file`: ROOT 表格的 CSV 檔案路徑
- `--output`: 輸出 CSV 檔案路徑（預設：triplet_dataset.csv）
- `--embedding_dim`: 文字 Embedding 維度（預設：512）
- `--max_text_length`: 文字最大長度（預設：100）
- `--triplet_per_anchor`: 每個 anchor 的 triplet 數量（預設：3）
- `--max_triplets`: 最大 triplet 數量（預設：10000）
- `--use_clip`: 使用 CLIP 模型（預設：True）
- `--no_clip`: 不使用 CLIP 模型，使用隨機向量

## 輸出檔案格式

### 1. BM25 樣本 (JSON)

```json
{
  "k": 5,
  "sample_size": 1000,
  "positive_samples": [[leaf_idx, root_idx], ...],
  "negative_samples": [[leaf_idx, root_idx], ...],
  "total_positive": 100,
  "total_negative": 500
}
```

### 2. 分類訓練集 (CSV)

| 欄位                | 說明                              | 範例                               |
| ------------------- | --------------------------------- | ---------------------------------- |
| `label`             | 正負樣本標籤 (1=正樣本, 0=負樣本) | 1                                  |
| `product_name_a`    | 商品 A 名稱                       | "22325999921474 22244642068224..." |
| `product_name_b`    | 商品 B 名稱                       | "21837616368731 21837616368731..." |
| `text_embedding_a`  | 商品 A 文字 embedding             | [0.1, 0.2, 0.3, ...]               |
| `text_embedding_b`  | 商品 B 文字 embedding             | [0.4, 0.5, 0.6, ...]               |
| `image_embedding_a` | 商品 A 圖片 embedding             | [0.7, 0.8, 0.9, ...]               |
| `image_embedding_b` | 商品 B 圖片 embedding             | [0.1, 0.2, 0.3, ...]               |

### 3. Triplet 訓練集 (CSV)

| 欄位                       | 說明                    | 範例                               |
| -------------------------- | ----------------------- | ---------------------------------- |
| `anchor_text`              | Anchor 文字             | "22325999921474 22244642068224..." |
| `positive_text`            | Positive 文字           | "21837616368731 21837616368731..." |
| `negative_text`            | Negative 文字           | "22313317189044 22238398409932..." |
| `anchor_text_embedding`    | Anchor 文字 embedding   | [0.1, 0.2, 0.3, ...]               |
| `positive_text_embedding`  | Positive 文字 embedding | [0.4, 0.5, 0.6, ...]               |
| `negative_text_embedding`  | Negative 文字 embedding | [0.7, 0.8, 0.9, ...]               |
| `anchor_image_embedding`   | Anchor 圖片 embedding   | [0.1, 0.2, 0.3, ...]               |
| `positive_image_embedding` | Positive 圖片 embedding | [0.4, 0.5, 0.6, ...]               |
| `negative_image_embedding` | Negative 圖片 embedding | [0.7, 0.8, 0.9, ...]               |

## 使用範例

### 載入分類訓練集

```python
import pandas as pd
import pickle
import numpy as np

# 載入資料
df = pd.read_csv('classification_dataset.csv')

# 載入向量化器
with open('classification_dataset_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 轉換 embedding 欄位
df['text_embedding_a'] = df['text_embedding_a'].apply(eval)
df['text_embedding_b'] = df['text_embedding_b'].apply(eval)
df['image_embedding_a'] = df['image_embedding_a'].apply(eval)
df['image_embedding_b'] = df['image_embedding_b'].apply(eval)

# 轉換為 numpy 陣列
X_text_a = np.array(df['text_embedding_a'].tolist())
X_text_b = np.array(df['text_embedding_b'].tolist())
X_image_a = np.array(df['image_embedding_a'].tolist())
X_image_b = np.array(df['image_embedding_b'].tolist())
y = df['label'].values

print(f"資料集大小: {len(df)}")
print(f"正樣本數量: {sum(y == 1)}")
print(f"負樣本數量: {sum(y == 0)}")
print(f"Embedding 維度: {X_text_a.shape[1]}")
```

### 載入 Triplet 訓練集

```python
import pandas as pd
import pickle
import numpy as np

# 載入資料
df = pd.read_csv('triplet_dataset.csv')

# 載入向量化器
with open('triplet_dataset_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 轉換 embedding 欄位
embedding_columns = [
    'anchor_text_embedding', 'positive_text_embedding', 'negative_text_embedding',
    'anchor_image_embedding', 'positive_image_embedding', 'negative_image_embedding'
]

for col in embedding_columns:
    df[col] = df[col].apply(eval)

# 轉換為 numpy 陣列
anchor_text = np.array(df['anchor_text_embedding'].tolist())
positive_text = np.array(df['positive_text_embedding'].tolist())
negative_text = np.array(df['negative_text_embedding'].tolist())
anchor_image = np.array(df['anchor_image_embedding'].tolist())
positive_image = np.array(df['positive_image_embedding'].tolist())
negative_image = np.array(df['negative_image_embedding'].tolist())

print(f"Triplet 數量: {len(df)}")
print(f"Embedding 維度: {anchor_text.shape[1]}")
```

## 注意事項

1. **圖片 Embedding**: 使用 OpenAI CLIP 模型生成高品質圖片 embedding，支援多種預訓練模型（ViT-B/32, ViT-L/14 等）

2. **文字 Embedding**: 使用 TF-IDF 向量化，可根據需要替換為其他方法（如 Word2Vec、BERT 等）

3. **資料正規化**: 所有 embedding 都已正規化，確保向量長度為 1

4. **記憶體使用**: 大規模資料集可能需要較多記憶體，建議分批處理

5. **檔案路徑**: 確保所有輸入檔案路徑正確，且檔案存在

6. **網路連接**: 使用 CLIP 功能時需要穩定的網路連接下載圖片

7. **GPU 支援**: CLIP 模型支援 GPU 加速，可大幅提升處理速度

## 故障排除

### 常見問題

1. **JSON 序列化錯誤**: 確保 numpy 整數類型已轉換為 Python 原生類型
2. **記憶體不足**: 減少 `sample_size` 或 `embedding_dim` 參數
3. **檔案不存在**: 檢查輸入檔案路徑是否正確
4. **權限錯誤**: 確保有寫入輸出目錄的權限

### 效能優化

1. **並行處理**: 對於大規模資料，可以考慮使用多進程處理
2. **批次處理**: 將大資料集分批處理以減少記憶體使用
3. **快取**: 對於重複使用的 embedding，可以考慮快取機制

## 進階用法

### 自定義 Embedding 模型

```python
# 替換圖片 embedding 生成函數
def custom_image_embedding(image_urls):
    # 使用真實的圖片 embedding 模型
    # 例如：ResNet, ViT, CLIP 等
    pass

# 替換文字 embedding 生成函數
def custom_text_embedding(texts):
    # 使用真實的文字 embedding 模型
    # 例如：BERT, Word2Vec, Sentence-BERT 等
    pass
```

### CLIP 模型配置

```python
from clip_image_processor import CLIPImageProcessor

# 使用不同的 CLIP 模型
processor = CLIPImageProcessor(model_name="ViT-L/14")  # 更大的模型，更好的效果
processor = CLIPImageProcessor(model_name="ViT-B/32")  # 較小的模型，更快的速度

# 指定設備
processor = CLIPImageProcessor(device="cuda")  # 使用 GPU
processor = CLIPImageProcessor(device="cpu")   # 使用 CPU
```

### 自定義相似度計算

```python
# 替換 BM25 相似度計算
def custom_similarity(leaf_features, root_features):
    # 使用自定義的相似度計算方法
    # 例如：餘弦相似度、歐幾里得距離等
    pass
```
