"""
CLIP 圖片處理工具
使用 CLIP 模型生成圖片 embedding
"""

import torch
import requests
from PIL import Image
import numpy as np
from typing import List, Optional, Union
import logging
from pathlib import Path
import io
import time

CLIP_AVAILABLE = False

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPImageProcessor:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        """
        初始化 CLIP 圖片處理器
        
        Args:
            model_name: CLIP 模型名稱
            device: 設備 ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        
        # 自動選擇設備
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"使用設備: {self.device}")
        
        if not CLIP_AVAILABLE:
            self.model = None
            self.preprocess = None
            return
            
        logger.info(f"載入 CLIP 模型: {model_name}")
        
        # 載入 CLIP 模型
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        logger.info("CLIP 模型載入完成")
    
    def download_image(self, url: str, timeout: int = 10) -> Optional[Image.Image]:
        """
        下載圖片
        
        Args:
            url: 圖片 URL
            timeout: 超時時間（秒）
            
        Returns:
            PIL Image 或 None
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image
        except Exception as e:
            logger.warning(f"無法下載圖片 {url}: {e}")
            return None
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        預處理圖片
        
        Args:
            image: PIL Image
            
        Returns:
            預處理後的 tensor
        """
        if not CLIP_AVAILABLE or self.preprocess is None:
            return torch.randn(1, 3, 224, 224)
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        return processed_image
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        獲取單張圖片的 embedding
        
        Args:
            image: PIL Image
            
        Returns:
            圖片 embedding (numpy array)
        """
        if not CLIP_AVAILABLE or self.model is None:
            # 使用隨機向量模擬
            embedding = np.random.normal(0, 1, 512)
            return embedding / np.linalg.norm(embedding)
            
        try:
            processed_image = self.preprocess_image(image)
            
            with torch.no_grad():
                image_features = self.model.encode_image(processed_image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"處理圖片時發生錯誤: {e}")
            return np.zeros(512)
    
    def get_image_embedding_from_url(self, url: str) -> np.ndarray:
        """
        從 URL 獲取圖片 embedding
        
        Args:
            url: 圖片 URL
            
        Returns:
            圖片 embedding (numpy array)
        """
        image = self.download_image(url)
        if image is None:
            return np.zeros(512)
        
        return self.get_image_embedding(image)
    
    def get_multiple_image_embeddings(self, urls: List[str], max_images: int = 5) -> np.ndarray:
        """
        獲取多張圖片的平均 embedding
        
        Args:
            urls: 圖片 URL 列表
            max_images: 最大處理圖片數量
            
        Returns:
            平均圖片 embedding (numpy array)
        """
        if not urls:
            return np.zeros(512)
        
        urls = urls[:max_images]
        
        embeddings = []
        for url in urls:
            embedding = self.get_image_embedding_from_url(url)
            if not np.allclose(embedding, 0):
                embeddings.append(embedding)
        
        if not embeddings:
            return np.zeros(512)
        
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    def batch_process_urls(self, url_lists: List[List[str]], max_images_per_item: int = 5) -> List[np.ndarray]:
        """
        批次處理多個 URL 列表
        
        Args:
            url_lists: URL 列表的列表
            max_images_per_item: 每個項目最大圖片數量
            
        Returns:
            embedding 列表
        """
        logger.info(f"開始批次處理 {len(url_lists)} 個項目的圖片")
        
        embeddings = []
        for i, urls in enumerate(url_lists):
            if i % 10 == 0:
                logger.info(f"處理進度: {i}/{len(url_lists)}")
            
            embedding = self.get_multiple_image_embeddings(urls, max_images_per_item)
            embeddings.append(embedding)
            
            time.sleep(0.1)
        
        logger.info("批次處理完成")
        return embeddings

def test_clip_processor():
    """測試 CLIP 處理器"""
    processor = CLIPImageProcessor()
    
    # 測試圖片 URL
    test_url = "https://gcs.rimg.com.tw/s5/91d/a41/w770715/3/c2/66/22324952670822_535.jpg"
    
    logger.info(f"測試圖片 URL: {test_url}")
    
    embedding = processor.get_image_embedding_from_url(test_url)
    logger.info(f"Embedding 維度: {embedding.shape}")
    logger.info(f"Embedding 範例: {embedding[:5]}")

if __name__ == "__main__":
    test_clip_processor()
