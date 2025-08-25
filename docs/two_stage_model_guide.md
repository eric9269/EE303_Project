# 兩階段商品匹配模型使用指南

本指南詳細說明如何使用兩階段商品匹配模型進行訓練和測試。

## 模型架構概述

我們的兩階段商品匹配系統包含兩個主要組件：

### 第一階段：相似度模型 (Triplet)

- **目的**: 學習商品間的相似度表示
- **輸入**: 商品文字和圖片 embedding
- **輸出**: 128 維相似度 embedding
- **損失函數**: Triplet Loss
- **作用**: 將商品映射到相似度空間，用於初步篩選

### 第二階段：分類模型 (Classification)

- **目的**: 判斷兩個商品是否匹配
- **輸入**: 兩個商品的文字和圖片 embedding
- **輸出**: 二分類結果 (0: 不匹配, 1: 匹配)
- **損失函數**: Cross Entropy Loss
- **作用**: 對通過第一階段的候選對進行精確分類

## 快速開始

### 1. 準備訓練資料

首先使用資料生成工具創建訓練資料：

```bash
cd tools
python generate_training_datasets.py \
    --leaf_file ../data/correct_structure_data/leaf_v1_correct.csv \
    --root_file ../data/correct_structure_data/root_v1_correct.csv \
    --output_dir training_data \
    --k 5 \
    --sample_size 1000 \
    --embedding_dim 512 \
    --use_clip
```

### 2. 執行完整管道

使用整合管道自動執行所有步驟：

```bash
python train_and_test_pipeline.py \
    --training_data_dir training_data \
    --output_dir pipeline_results \
    --text_embedding_dim 512 \
    --image_embedding_dim 512 \
    --hidden_dim 256 \
    --output_dim 128 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --similarity_threshold 0.5
```

## 詳細使用說明

### 單獨訓練 Triplet 模型

```bash
python train_triplet_model.py \
    --data_path training_data/triplet_dataset.csv \
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

**參數說明**:

- `--data_path`: Triplet 訓練資料路徑
- `--output_dir`: 模型輸出目錄
- `--text_embedding_dim`: 文字 embedding 維度
- `--image_embedding_dim`: 圖片 embedding 維度
- `--hidden_dim`: 隱藏層維度
- `--output_dim`: 輸出維度
- `--batch_size`: 批次大小
- `--epochs`: 訓練輪數
- `--learning_rate`: 學習率
- `--device`: 設備 (auto/cpu/cuda)

### 單獨訓練 Classification 模型

```bash
python train_classification_model.py \
    --data_path training_data/classification_dataset.csv \
    --output_dir classification_model \
    --text_embedding_dim 512 \
    --image_embedding_dim 512 \
    --hidden_dim 256 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001 \
    --device auto
```

**參數說明**:

- `--data_path`: Classification 訓練資料路徑
- `--output_dir`: 模型輸出目錄
- `--text_embedding_dim`: 文字 embedding 維度
- `--image_embedding_dim`: 圖片 embedding 維度
- `--hidden_dim`: 隱藏層維度
- `--batch_size`: 批次大小
- `--epochs`: 訓練輪數
- `--learning_rate`: 學習率
- `--device`: 設備 (auto/cpu/cuda)

### 測試模型

```bash
python test_models.py \
    --test_data_path training_data/classification_dataset.csv \
    --similarity_model_path triplet_model/best_triplet_model.pth \
    --classification_model_path classification_model/best_classification_model.pth \
    --output_dir test_results \
    --similarity_threshold 0.5 \
    --batch_size 32 \
    --device auto
```

**參數說明**:

- `--test_data_path`: 測試資料路徑
- `--similarity_model_path`: 相似度模型路徑
- `--classification_model_path`: 分類模型路徑
- `--output_dir`: 測試結果輸出目錄
- `--similarity_threshold`: 相似度閾值
- `--batch_size`: 批次大小
- `--device`: 設備 (auto/cpu/cuda)

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

## 自定義文字 Embedding

### 使用 BERT

```python
from text_embedding_processor import create_text_processor

# 創建 BERT 處理器
bert_processor = create_text_processor(
    processor_type="bert",
    model_name="bert-base-chinese",
    embedding_dim=768
)

# 獲取 embedding
text = "商品名稱"
embedding = bert_processor.get_embedding(text)
```

### 使用自定義模型

```python
from text_embedding_processor import CustomTextProcessor

class MyCustomTextProcessor(CustomTextProcessor):
    def __init__(self, embedding_dim=512, device="auto", model_path=None):
        super().__init__(embedding_dim, device, model_path)
        # 載入你的自定義模型
        self.load_custom_model(model_path)

    def get_embedding(self, text: str) -> np.ndarray:
        # 實現你的自定義邏輯
        # 例如：使用你的預訓練模型進行推理
        return super().get_embedding(text)

# 使用自定義處理器
custom_processor = MyCustomTextProcessor(
    embedding_dim=512,
    model_path="path/to/your/model.pth"
)
```

## 兩階段推理流程

### 1. 載入模型

```python
from test_models import ModelTester

# 創建測試器
tester = ModelTester()

# 載入模型
similarity_model = tester.load_similarity_model('triplet_model/best_triplet_model.pth')
classification_model = tester.load_classification_model('classification_model/best_classification_model.pth')
```

### 2. 進行預測

```python
def predict_match(text_a, image_a, text_b, image_b, similarity_threshold=0.5):
    """
    預測兩個商品是否匹配

    Args:
        text_a: 商品 A 的文字 embedding
        image_a: 商品 A 的圖片 embedding
        text_b: 商品 B 的文字 embedding
        image_b: 商品 B 的圖片 embedding
        similarity_threshold: 相似度閾值

    Returns:
        is_match: 是否匹配
        confidence: 置信度
    """
    # 第一階段：相似度計算
    similarity_embedding_a = similarity_model(text_a, image_a)
    similarity_embedding_b = similarity_model(text_b, image_b)
    similarity = torch.cosine_similarity(similarity_embedding_a, similarity_embedding_b, dim=0)

    # 檢查是否通過第一階段
    if similarity < similarity_threshold:
        return False, 0.0

    # 第二階段：分類
    outputs = classification_model(text_a, text_b, image_a, image_b)
    probabilities = torch.softmax(outputs, dim=0)
    prediction = torch.argmax(probabilities).item()
    confidence = probabilities[prediction].item()

    return prediction == 1, confidence
```

## 參數調優指南

### 相似度閾值調優

相似度閾值影響兩階段模型的性能平衡：

- **較低閾值 (0.3-0.5)**: 更多候選對進入第二階段，可能提高召回率但降低精確率
- **較高閾值 (0.6-0.8)**: 更少候選對進入第二階段，可能提高精確率但降低召回率

建議根據實際需求調整：

```python
# 高精確率設置
similarity_threshold = 0.7

# 高召回率設置
similarity_threshold = 0.3

# 平衡設置
similarity_threshold = 0.5
```

### 模型架構調優

可以調整以下參數來優化模型性能：

```python
# 更大的模型（需要更多計算資源）
hidden_dim = 512
output_dim = 256

# 更小的模型（更快的推理速度）
hidden_dim = 128
output_dim = 64

# 平衡設置
hidden_dim = 256
output_dim = 128
```

### 訓練參數調優

```python
# 更長時間訓練（可能獲得更好性能）
epochs = 100
learning_rate = 0.0005

# 快速訓練（用於原型開發）
epochs = 20
learning_rate = 0.001

# 平衡設置
epochs = 50
learning_rate = 0.001
```

## 性能評估

### 單階段模型評估

```python
# 相似度模型評估
similarity_results = tester.test_similarity_model(
    similarity_model, test_data_path, batch_size=32
)

print(f"相似度模型準確率: {similarity_results['accuracy']:.4f}")
print(f"平均正樣本相似度: {similarity_results['avg_positive_similarity']:.4f}")
print(f"平均負樣本相似度: {similarity_results['avg_negative_similarity']:.4f}")

# 分類模型評估
classification_results = tester.test_classification_model(
    classification_model, test_data_path, batch_size=32
)

print(f"分類模型準確率: {classification_results['accuracy']:.4f}")
print(f"F1 分數: {classification_results['f1']:.4f}")
```

### 兩階段模型評估

```python
# 兩階段模型評估
two_stage_results = tester.test_two_stage_model(
    similarity_model, classification_model, test_data_path,
    similarity_threshold=0.5, batch_size=32
)

print(f"兩階段模型準確率: {two_stage_results['accuracy']:.4f}")
print(f"第一階段通過率: {two_stage_results['stage1_pass_rate']:.4f}")
print(f"最終 F1 分數: {two_stage_results['f1']:.4f}")
```

## 故障排除

### 常見問題

1. **記憶體不足**

   - 減少 `batch_size`
   - 減少 `hidden_dim` 和 `output_dim`
   - 使用 CPU 訓練

2. **訓練不收斂**

   - 調整 `learning_rate`
   - 增加 `epochs`
   - 檢查資料品質

3. **過擬合**

   - 增加 Dropout 率
   - 減少模型複雜度
   - 使用更多訓練資料

4. **相似度閾值選擇困難**
   - 使用驗證集調優
   - 根據業務需求平衡精確率和召回率

### 效能優化

1. **GPU 加速**

   ```bash
   --device cuda
   ```

2. **批次處理**

   ```bash
   --batch_size 64  # 根據 GPU 記憶體調整
   ```

3. **混合精度訓練**
   ```python
   # 在訓練代碼中添加
   scaler = torch.cuda.amp.GradScaler()
   ```

## 未來改進方向

1. **模型架構改進**

   - 嘗試 Transformer 架構
   - 使用注意力機制
   - 實現多尺度特徵融合

2. **損失函數改進**

   - 使用 Focal Loss
   - 實現對比學習
   - 添加正則化項

3. **資料增強**

   - 文字增強技術
   - 圖片增強技術
   - 對抗訓練

4. **集成學習**

   - 多模型集成
   - 投票機制
   - 堆疊集成

5. **線上學習**
   - 增量學習
   - 持續學習
   - 適應性更新
