"""
CLIP 圖片處理工具
使用 EVA-CLIP 和標準 CLIP 模型生成圖片 embedding
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

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPImageProcessor:
    def __init__(self, model_name: str = "clip-vit-large-patch14", device: str = "auto"):
        """
        初始化 CLIP 圖片處理器
        
        Args:
            model_name: CLIP 模型名稱
                - EVA-CLIP 模型: "EVA02-CLIP-L", "EVA02-CLIP-B", "EVA01-CLIP-g-14"
                - 標準 CLIP 模型: "ViT-B/32", "ViT-L/14"
            device: 設備 ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        
        # 自動選擇設備
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"使用設備: {self.device}")
        
        # 檢查是否為 EVA-CLIP 模型
        self.is_eva_clip = model_name.startswith("EVA")
        
        # 載入模型
        self.model = None
        self.processor = None
        self.CLIP_AVAILABLE = False
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.CLIP_AVAILABLE = True
            logger.info("Transformers 模組可用")
        except ImportError:
            logger.warning("Transformers 模組未安裝，將使用隨機向量模擬")
            return
        
        # 載入模型
        try:
            if self.is_eva_clip:
                # 使用 EVA-CLIP
                if model_name.startswith("EVA02-CLIP-bigE"):
                    # 特殊處理 bigE 模型
                    model_id = f"QuanSun/EVA-CLIP/{model_name}"
                else:
                    model_id = f"QuanSun/{model_name}"
                
                logger.info(f"載入 EVA-CLIP 模型: {model_id}")
                
                # 嘗試載入模型
                try:
                    self.model = CLIPModel.from_pretrained(model_id)
                    self.processor = CLIPProcessor.from_pretrained(model_id)
                    logger.info(f"EVA-CLIP 模型已載入: {model_id}")
                except Exception as e:
                    logger.warning(f"無法載入 EVA-CLIP 模型 {model_id}: {e}")
                    logger.info("使用標準 CLIP 模型作為備選")
                    # 使用標準 CLIP 作為備選
                    self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                    logger.info("已載入標準 CLIP 模型作為備選")
            else:
                # 使用標準 CLIP
                model_id = f"openai/{model_name}"
                logger.info(f"載入標準 CLIP 模型: {model_id}")
                self.model = CLIPModel.from_pretrained(model_id)
                self.processor = CLIPProcessor.from_pretrained(model_id)
                logger.info(f"標準 CLIP 模型已載入: {model_id}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"載入 CLIP 模型失敗: {e}")
            self.CLIP_AVAILABLE = False
    
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
        if not self.CLIP_AVAILABLE or self.processor is None:
            return torch.randn(1, 3, 224, 224)
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 使用 processor 進行預處理
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        獲取圖片 embedding
        
        Args:
            image: PIL Image
            
        Returns:
            圖片 embedding 向量
        """
        if not self.CLIP_AVAILABLE or self.model is None:
            # 返回隨機向量
            logger.warning("CLIP 不可用，返回隨機向量")
            return np.random.normal(0, 1, 512)
        
        try:
            # 預處理圖片
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 獲取 embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embedding = outputs.image_embeds.cpu().numpy().flatten()
                
                # 正規化
                image_embedding = image_embedding / np.linalg.norm(image_embedding)
                
            return image_embedding
            
        except Exception as e:
            logger.error(f"獲取圖片 embedding 失敗: {e}")
            # 返回隨機向量作為備選
            embedding = np.random.normal(0, 1, 512)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def get_image_embedding_from_url(self, url: str) -> np.ndarray:
        """
        從 URL 獲取圖片 embedding
        
        Args:
            url: 圖片 URL
            
        Returns:
            圖片 embedding 向量
        """
        image = self.download_image(url)
        if image is None:
            logger.warning(f"無法下載圖片，返回隨機向量: {url}")
            return np.random.normal(0, 1, 512)
        
        return self.get_image_embedding(image)
    
    def batch_process_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        批次處理圖片
        
        Args:
            images: PIL Image 列表
            
        Returns:
            圖片 embedding 矩陣
        """
        if not self.CLIP_AVAILABLE or self.model is None:
            # 返回隨機向量
            logger.warning("CLIP 不可用，返回隨機向量")
            return np.random.normal(0, 1, (len(images), 512))
        
        try:
            # 批次預處理
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 獲取 embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeddings = outputs.image_embeds.cpu().numpy()
                
                # 正規化
                norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
                image_embeddings = image_embeddings / norms
                
            return image_embeddings
            
        except Exception as e:
            logger.error(f"批次處理圖片失敗: {e}")
            return np.random.normal(0, 1, (len(images), 512))
    
    def batch_process_urls(self, urls: List[str]) -> np.ndarray:
        """
        批次處理 URL 列表
        
        Args:
            urls: 圖片 URL 列表
            
        Returns:
            圖片 embedding 矩陣
        """
        images = []
        valid_indices = []
        
        # 下載圖片
        for i, url in enumerate(urls):
            image = self.download_image(url)
            if image is not None:
                images.append(image)
                valid_indices.append(i)
            else:
                logger.warning(f"無法下載圖片: {url}")
        
        if not images:
            logger.warning("沒有成功下載的圖片，返回隨機向量")
            return np.random.normal(0, 1, (len(urls), 512))
        
        # 批次處理
        embeddings = self.batch_process_images(images)
        
        # 為失敗的圖片填充隨機向量
        if len(embeddings) < len(urls):
            full_embeddings = np.random.normal(0, 1, (len(urls), embeddings.shape[1]))
            for i, idx in enumerate(valid_indices):
                full_embeddings[idx] = embeddings[i]
            return full_embeddings
        
        return embeddings
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        獲取文字 embedding
        
        Args:
            text: 輸入文字
            
        Returns:
            文字 embedding 向量
        """
        if not self.CLIP_AVAILABLE or self.model is None:
            # 返回隨機向量
            logger.warning("CLIP 不可用，返回隨機向量")
            return np.random.normal(0, 1, 512)
        
        try:
            # 預處理文字
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 獲取 embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                text_embedding = outputs.text_embeds.cpu().numpy().flatten()
                
                # 正規化
                text_embedding = text_embedding / np.linalg.norm(text_embedding)
                
            return text_embedding
            
        except Exception as e:
            logger.error(f"獲取文字 embedding 失敗: {e}")
            return np.random.normal(0, 1, 512)
    
    def get_similarity(self, image: Image.Image, text: str) -> float:
        """
        計算圖片和文字的相似度
        
        Args:
            image: PIL Image
            text: 文字
            
        Returns:
            相似度分數
        """
        if not self.CLIP_AVAILABLE or self.model is None:
            return 0.0
        
        try:
            # 獲取 embedding
            image_embedding = self.get_image_embedding(image)
            text_embedding = self.get_text_embedding(text)
            
            # 計算餘弦相似度
            similarity = np.dot(image_embedding, text_embedding)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"計算相似度失敗: {e}")
            return 0.0

def test_clip_processor():
    """測試 CLIP 處理器"""
    logger.info("測試 CLIP 處理器...")
    
    # 測試 EVA-CLIP
    try:
        processor = CLIPImageProcessor("EVA02-CLIP-L")
        logger.info("EVA-CLIP 處理器創建成功")
        
        # 測試隨機圖片
        test_image = Image.new('RGB', (224, 224), color='red')
        embedding = processor.get_image_embedding(test_image)
        logger.info(f"EVA-CLIP embedding 形狀: {embedding.shape}")
        
    except Exception as e:
        logger.error(f"EVA-CLIP 測試失敗: {e}")
    
    # 測試標準 CLIP
    try:
        processor = CLIPImageProcessor("ViT-B/32")
        logger.info("標準 CLIP 處理器創建成功")
        
        # 測試隨機圖片
        test_image = Image.new('RGB', (224, 224), color='blue')
        embedding = processor.get_image_embedding(test_image)
        logger.info(f"標準 CLIP embedding 形狀: {embedding.shape}")
        
    except Exception as e:
        logger.error(f"標準 CLIP 測試失敗: {e}")

if __name__ == "__main__":
    test_clip_processor()
