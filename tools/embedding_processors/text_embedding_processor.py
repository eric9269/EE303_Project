import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any
import logging
from pathlib import Path
import json

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextEmbeddingProcessor:
    """文字 Embedding 處理器基類"""
    
    def __init__(self, embedding_dim: int = 512, device: str = "auto"):
        """
        初始化文字處理器
        
        Args:
            embedding_dim: embedding 維度
            device: 設備
        """
        self.embedding_dim = embedding_dim
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"文字處理器使用設備: {self.device}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        獲取文字 embedding
        
        Args:
            text: 輸入文字
            
        Returns:
            embedding 向量
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        批次獲取文字 embedding
        
        Args:
            texts: 文字列表
            
        Returns:
            embedding 矩陣
        """
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)

class BERTTextProcessor(TextEmbeddingProcessor):
    """BERT 文字處理器"""
    
    def __init__(self, model_name: str = "bert-base-chinese", embedding_dim: int = 768, device: str = "auto"):
        """
        初始化 BERT 處理器
        
        Args:
            model_name: BERT 模型名稱
            embedding_dim: embedding 維度
            device: 設備
        """
        super().__init__(embedding_dim, device)
        
        try:
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"BERT 模型已載入: {model_name}")
            
        except ImportError:
            logger.error("transformers 模組未安裝，請執行: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"載入 BERT 模型失敗: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        獲取 BERT embedding
        
        Args:
            text: 輸入文字
            
        Returns:
            BERT embedding 向量
        """
        # 標記化
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 移動到設備
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向傳播
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 使用 [CLS] token 的 embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # 正規化
            embedding = embedding / np.linalg.norm(embedding)
            
            # 調整維度
            if len(embedding) != self.embedding_dim:
                if len(embedding) > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
                else:
                    # 填充到目標維度
                    padding = np.zeros(self.embedding_dim - len(embedding))
                    embedding = np.concatenate([embedding, padding])
        
        return embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        批次獲取 BERT embedding
        
        Args:
            texts: 文字列表
            
        Returns:
            BERT embedding 矩陣
        """
        # 標記化
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 移動到設備
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向傳播
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 使用 [CLS] token 的 embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 正規化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # 調整維度
            if embeddings.shape[1] != self.embedding_dim:
                if embeddings.shape[1] > self.embedding_dim:
                    embeddings = embeddings[:, :self.embedding_dim]
                else:
                    # 填充到目標維度
                    padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                    embeddings = np.concatenate([embeddings, padding], axis=1)
        
        return embeddings

class CustomTextProcessor(TextEmbeddingProcessor):
    """自定義文字處理器（預留接口）"""
    
    def __init__(self, embedding_dim: int = 512, device: str = "auto", model_path: str = None):
        """
        初始化自定義處理器
        
        Args:
            embedding_dim: embedding 維度
            device: 設備
            model_path: 自定義模型路徑
        """
        super().__init__(embedding_dim, device)
        self.model_path = model_path
        self.model = None
        
        logger.info("自定義文字處理器已初始化，請實現 get_embedding 方法")
    
    def load_custom_model(self, model_path: str):
        """
        載入自定義模型
        
        Args:
            model_path: 模型路徑
        """
        # 這裡可以載入你的自定義模型
        # 例如：
        # self.model = torch.load(model_path, map_location=self.device)
        # self.model.eval()
        
        logger.info(f"自定義模型載入接口: {model_path}")
        logger.info("請在子類中實現具體的模型載入邏輯")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        獲取自定義 embedding
        
        Args:
            text: 輸入文字
            
        Returns:
            自定義 embedding 向量
        """
        # 這裡應該實現你的自定義模型推理邏輯
        # 目前返回隨機向量作為示例
        
        logger.warning("使用隨機向量作為自定義 embedding 的示例")
        
        embedding = np.random.normal(0, 1, self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

class TFIDFTextProcessor(TextEmbeddingProcessor):
    """TF-IDF 文字處理器（作為備用）"""
    
    def __init__(self, embedding_dim: int = 512, device: str = "auto"):
        """
        初始化 TF-IDF 處理器
        
        Args:
            embedding_dim: embedding 維度
            device: 設備
        """
        super().__init__(embedding_dim, device)
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=embedding_dim,
                stop_words=None,
                ngram_range=(1, 2)
            )
            
            logger.info("TF-IDF 向量化器已初始化")
            
        except ImportError:
            logger.error("scikit-learn 模組未安裝")
            raise
    
    def fit(self, texts: List[str]):
        """
        擬合 TF-IDF 向量化器
        
        Args:
            texts: 訓練文字列表
        """
        self.vectorizer.fit(texts)
        logger.info("TF-IDF 向量化器已擬合")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        獲取 TF-IDF embedding
        
        Args:
            text: 輸入文字
            
        Returns:
            TF-IDF embedding 向量
        """
        embedding = self.vectorizer.transform([text]).toarray().flatten()
        
        # 正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # 調整維度
        if len(embedding) != self.embedding_dim:
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                padding = np.zeros(self.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
        
        return embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        批次獲取 TF-IDF embedding
        
        Args:
            texts: 文字列表
            
        Returns:
            TF-IDF embedding 矩陣
        """
        embeddings = self.vectorizer.transform(texts).toarray()
        
        # 正規化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # 調整維度
        if embeddings.shape[1] != self.embedding_dim:
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            else:
                padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, padding], axis=1)
        
        return embeddings

def create_text_processor(processor_type: str = "bert", **kwargs) -> TextEmbeddingProcessor:
    """
    創建文字處理器工廠函數
    
    Args:
        processor_type: 處理器類型 ("bert", "custom", "tfidf")
        **kwargs: 其他參數
        
    Returns:
        文字處理器實例
    """
    if processor_type.lower() == "bert":
        return BERTTextProcessor(**kwargs)
    elif processor_type.lower() == "custom":
        return CustomTextProcessor(**kwargs)
    elif processor_type.lower() == "tfidf":
        return TFIDFTextProcessor(**kwargs)
    else:
        raise ValueError(f"不支援的處理器類型: {processor_type}")

def test_text_processor():
    """測試文字處理器"""
    logger.info("測試文字處理器...")
    
    test_texts = [
        "這是一個測試文字",
        "另一個測試文字",
        "商品名稱測試"
    ]
    
    # 測試 BERT 處理器
    try:
        bert_processor = create_text_processor("bert", embedding_dim=512)
        bert_embeddings = bert_processor.get_batch_embeddings(test_texts)
        logger.info(f"BERT embeddings 形狀: {bert_embeddings.shape}")
    except Exception as e:
        logger.warning(f"BERT 處理器測試失敗: {e}")
    
    # 測試 TF-IDF 處理器
    try:
        tfidf_processor = create_text_processor("tfidf", embedding_dim=512)
        tfidf_processor.fit(test_texts)
        tfidf_embeddings = tfidf_processor.get_batch_embeddings(test_texts)
        logger.info(f"TF-IDF embeddings 形狀: {tfidf_embeddings.shape}")
    except Exception as e:
        logger.warning(f"TF-IDF 處理器測試失敗: {e}")
    
    # 測試自定義處理器
    try:
        custom_processor = create_text_processor("custom", embedding_dim=512)
        custom_embeddings = custom_processor.get_batch_embeddings(test_texts)
        logger.info(f"Custom embeddings 形狀: {custom_embeddings.shape}")
    except Exception as e:
        logger.warning(f"Custom 處理器測試失敗: {e}")

if __name__ == "__main__":
    test_text_processor()
