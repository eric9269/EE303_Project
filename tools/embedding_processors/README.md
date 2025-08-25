# Embedding 處理器

此目錄包含用於生成文字和圖片 embedding 的工具。

## 檔案說明

### `clip_image_processor.py`

- **功能**: CLIP 圖片處理器
- **用途**: 使用 OpenAI CLIP 模型生成圖片 embedding
- **特點**:
  - 支援多種 CLIP 模型 (ViT-B/32, ViT-L/14 等)
  - 自動下載和處理圖片
  - 批次處理支援
  - 錯誤處理和降級機制

### `text_embedding_processor.py`

- **功能**: 文字 Embedding 處理器
- **用途**: 生成文字 embedding，支援多種方法
- **支援的方法**:
  - **BERT**: 使用預訓練的 BERT 模型
  - **TF-IDF**: 傳統的 TF-IDF 向量化
  - **自定義**: 預留接口，可替換為自定義模型

## 使用方式

### CLIP 圖片處理

```python
from clip_image_processor import CLIPImageProcessor

# 創建處理器
processor = CLIPImageProcessor(model_name="ViT-B/32")

# 處理單張圖片
embedding = processor.get_image_embedding_from_url("https://example.com/image.jpg")

# 批次處理
urls = ["url1", "url2", "url3"]
embeddings = processor.batch_process_urls(urls)
```

### 文字處理

```python
from text_embedding_processor import create_text_processor

# 使用 BERT
bert_processor = create_text_processor("bert", model_name="bert-base-chinese")
embedding = bert_processor.get_embedding("商品名稱")

# 使用 TF-IDF
tfidf_processor = create_text_processor("tfidf")
tfidf_processor.fit(["訓練文字1", "訓練文字2"])
embedding = tfidf_processor.get_embedding("測試文字")

# 使用自定義模型
custom_processor = create_text_processor("custom", model_path="path/to/model")
embedding = custom_processor.get_embedding("商品名稱")
```

## 自定義模型擴展

### 實現自定義文字處理器

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
```

## 配置選項

### CLIP 配置

- **模型選擇**: ViT-B/32 (快速), ViT-L/14 (高品質)
- **設備**: CPU, CUDA, 自動選擇
- **批次大小**: 可調整以平衡速度和記憶體使用

### 文字處理配置

- **BERT 模型**: bert-base-chinese, bert-base-multilingual 等
- **TF-IDF 參數**: max_features, ngram_range, stop_words
- **自定義模型**: 完全可配置的接口

## 注意事項

- CLIP 需要網路連接下載圖片
- BERT 模型首次使用會下載預訓練權重
- 建議使用 GPU 加速處理
- 所有 embedding 都會自動正規化
