"""
Triplet 訓練樣本生成工具
生成格式：[anchor text, positive text, negative text, anchor image embedding, positive image embedding, negative image embedding]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import List, Tuple, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import random
from tools.embedding_processors.clip_image_processor import CLIPImageProcessor

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripletDatasetGenerator:
    def __init__(self, 
                 embedding_dim: int = 512, 
                 max_text_length: int = 10240,
                 triplet_per_anchor: int = 3,
                 use_clip: bool = True):
        """
        初始化 Triplet 資料集生成器
        
        Args:
            embedding_dim: 文字 embedding 維度
            max_text_length: 文字最大長度
            triplet_per_anchor: 每個 anchor 生成的 triplet 數量
            use_clip: 是否使用 CLIP 模型
        """
        self.embedding_dim = embedding_dim
        self.max_text_length = max_text_length
        self.triplet_per_anchor = triplet_per_anchor
        self.use_clip = use_clip
        self.vectorizer = None
        self.clip_processor = None
        
        if self.use_clip:
            self.clip_processor = CLIPImageProcessor()
            if hasattr(self.clip_processor, 'model') and self.clip_processor.model:
                self.clip_embedding_dim = self.clip_processor.model.visual.output_dim
            else:
                self.clip_embedding_dim = 512
        
    def load_samples(self, samples_file: str) -> Dict[str, Any]:
        """
        載入正負樣本
        
        Args:
            samples_file: 樣本檔案路徑
            
        Returns:
            樣本資料字典
        """
        logger.info(f"載入樣本資料: {samples_file}")
        
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples_data = json.load(f)
        
        return samples_data
    
    def load_data(self, leaf_file: str, root_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        載入 LEAF 和 ROOT 資料
        
        Args:
            leaf_file: LEAF CSV 檔案路徑
            root_file: ROOT CSV 檔案路徑
            
        Returns:
            leaf_data, root_data
        """
        logger.info(f"載入 LEAF 資料: {leaf_file}")
        leaf_data = pd.read_csv(leaf_file, encoding='utf-8-sig')
        
        logger.info(f"載入 ROOT 資料: {root_file}")
        root_data = pd.read_csv(root_file, encoding='utf-8-sig')
        
        return leaf_data, root_data
    
    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        生成文字 embedding
        
        Args:
            texts: 文字列表
            
        Returns:
            文字 embedding 矩陣
        """
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.embedding_dim,
                stop_words='english',
                ngram_range=(1, 2)
            )
            return self.vectorizer.fit_transform(texts).toarray()
        else:
            return self.vectorizer.transform(texts).toarray()
    
    def generate_image_embeddings(self, image_urls: List[List[str]]) -> np.ndarray:
        """
        生成圖片 embedding
        
        Args:
            image_urls: 圖片 URL 列表
            
        Returns:
            圖片 embedding 矩陣
        """
        if self.use_clip and self.clip_processor:
            logger.info("使用 CLIP 模型生成圖片 embedding...")
            embeddings = self.clip_processor.batch_process_urls(image_urls)
            return np.array(embeddings)
        else:
            logger.info("使用隨機向量模擬圖片 embedding...")
            embeddings = []
            
            for urls in image_urls:
                if urls:
                    embedding = np.random.normal(0, 1, self.embedding_dim)
                    embedding = embedding / np.linalg.norm(embedding)
                else:
                    embedding = np.zeros(self.embedding_dim)
                
                embeddings.append(embedding)
            
            return np.array(embeddings)
    
    def extract_product_name(self, row: pd.Series) -> str:
        """
        提取商品名稱
        
        Args:
            row: 資料行
            
        Returns:
            商品名稱
        """
        features = []
        
        # 產品ID
        if 'product_ids' in row and pd.notna(row['product_ids']):
            try:
                product_ids = json.loads(row['product_ids'])
                if product_ids:
                    features.extend([str(pid) for pid in product_ids[:3]])  # 取前3個
            except:
                pass
        
        # 商店名稱
        if 'shops' in row and pd.notna(row['shops']):
            try:
                shops = json.loads(row['shops'])
                if shops:
                    features.extend([str(shop) for shop in shops[:2]])  # 取前2個
            except:
                pass
        
        # 價格
        if 'prices' in row and pd.notna(row['prices']):
            try:
                prices = json.loads(row['prices'])
                if prices:
                    features.extend([str(price) for price in prices[:2]])  # 取前2個
            except:
                pass
        
        product_name = ' '.join(features)
        return product_name[:self.max_text_length]  # 限制長度
    
    def generate_triplets(self, 
                         positive_pairs: List[Tuple[int, int]],
                         negative_pairs: List[Tuple[int, int]],
                         leaf_names: List[str],
                         root_names: List[str],
                         leaf_text_embeddings: np.ndarray,
                         root_text_embeddings: np.ndarray,
                         leaf_image_embeddings: np.ndarray,
                         root_image_embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """
        生成 triplet 樣本
        
        Args:
            positive_pairs: 正樣本對
            negative_pairs: 負樣本對
            leaf_names: LEAF 商品名稱
            root_names: ROOT 商品名稱
            leaf_text_embeddings: LEAF 文字 embedding
            root_text_embeddings: ROOT 文字 embedding
            leaf_image_embeddings: LEAF 圖片 embedding
            root_image_embeddings: ROOT 圖片 embedding
            
        Returns:
            triplet 樣本列表
        """
        logger.info("生成 triplet 樣本...")
        
        triplets = []
        
        # 為每個正樣本生成 triplet
        for leaf_idx, root_idx in positive_pairs:
            anchor_text = leaf_names[leaf_idx]
            positive_text = root_names[root_idx]
            anchor_text_embedding = leaf_text_embeddings[leaf_idx]
            positive_text_embedding = root_text_embeddings[root_idx]
            anchor_image_embedding = leaf_image_embeddings[leaf_idx]
            positive_image_embedding = root_image_embeddings[root_idx]
            
            # 為每個 anchor 生成多個 triplet
            for _ in range(self.triplet_per_anchor):
                # 隨機選擇一個負樣本作為 negative
                negative_pair = random.choice(negative_pairs)
                negative_leaf_idx, negative_root_idx = negative_pair
                
                # 隨機決定 negative 是來自 leaf 還是 root
                if random.random() < 0.5:
                    negative_text = leaf_names[negative_leaf_idx]
                    negative_text_embedding = leaf_text_embeddings[negative_leaf_idx]
                    negative_image_embedding = leaf_image_embeddings[negative_leaf_idx]
                else:
                    negative_text = root_names[negative_root_idx]
                    negative_text_embedding = root_text_embeddings[negative_root_idx]
                    negative_image_embedding = root_image_embeddings[negative_root_idx]
                
                triplet = {
                    'anchor_text': anchor_text,
                    'positive_text': positive_text,
                    'negative_text': negative_text,
                    'anchor_text_embedding': anchor_text_embedding.tolist(),
                    'positive_text_embedding': positive_text_embedding.tolist(),
                    'negative_text_embedding': negative_text_embedding.tolist(),
                    'anchor_image_embedding': anchor_image_embedding.tolist(),
                    'positive_image_embedding': positive_image_embedding.tolist(),
                    'negative_image_embedding': negative_image_embedding.tolist(),
                    'anchor_leaf_idx': leaf_idx,
                    'positive_root_idx': root_idx,
                    'negative_leaf_idx': negative_leaf_idx,
                    'negative_root_idx': negative_root_idx
                }
                
                triplets.append(triplet)
        
        logger.info(f"生成了 {len(triplets)} 個 triplet 樣本")
        return triplets
    
    def generate_triplet_dataset(self, 
                                samples_file: str,
                                leaf_file: str, 
                                root_file: str,
                                output_file: str,
                                max_triplets: int = 10000) -> None:
        """
        生成 triplet 訓練集
        
        Args:
            samples_file: 樣本檔案路徑
            leaf_file: LEAF CSV 檔案路徑
            root_file: ROOT CSV 檔案路徑
            output_file: 輸出檔案路徑
            max_triplets: 最大 triplet 數量
        """
        logger.info("開始生成 triplet 訓練集...")
        
        # 載入資料
        samples_data = self.load_samples(samples_file)
        leaf_data, root_data = self.load_data(leaf_file, root_file)
        
        # 解析 JSON 欄位
        for df in [leaf_data, root_data]:
            for col in ['product_ids', 'image_urls', 'page_urls', 'prices', 'shops']:
                if col in df.columns:
                    df[f'{col}_parsed'] = df[col].apply(
                        lambda x: json.loads(x) if pd.notna(x) and x != '[]' else []
                    )
        
        # 生成商品名稱
        logger.info("生成商品名稱...")
        leaf_names = [self.extract_product_name(row) for _, row in leaf_data.iterrows()]
        root_names = [self.extract_product_name(row) for _, row in root_data.iterrows()]
        
        # 生成文字 embedding
        logger.info("生成文字 embedding...")
        all_names = leaf_names + root_names
        text_embeddings = self.generate_text_embeddings(all_names)
        leaf_text_embeddings = text_embeddings[:len(leaf_names)]
        root_text_embeddings = text_embeddings[len(leaf_names):]
        
        # 生成圖片 embedding
        logger.info("生成圖片 embedding...")
        leaf_image_urls = [row['image_urls_parsed'] for _, row in leaf_data.iterrows()]
        root_image_urls = [row['image_urls_parsed'] for _, row in root_data.iterrows()]
        
        leaf_image_embeddings = self.generate_image_embeddings(leaf_image_urls)
        root_image_embeddings = self.generate_image_embeddings(root_image_urls)
        
        # 生成 triplet 樣本
        triplets = self.generate_triplets(
            samples_data['positive_samples'],
            samples_data['negative_samples'],
            leaf_names,
            root_names,
            leaf_text_embeddings,
            root_text_embeddings,
            leaf_image_embeddings,
            root_image_embeddings
        )
        
        # 限制 triplet 數量
        if len(triplets) > max_triplets:
            triplets = random.sample(triplets, max_triplets)
            logger.info(f"限制 triplet 數量為: {max_triplets}")
        
        # 轉換為 DataFrame
        df = pd.DataFrame(triplets)
        
        # 保存資料
        logger.info(f"保存 triplet 訓練集: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 保存向量化器
        vectorizer_file = output_file.replace('.csv', '_vectorizer.pkl')
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # 統計資訊
        logger.info(f"Triplet 訓練集生成完成:")
        logger.info(f"  總 triplet 數: {len(df)}")
        logger.info(f"  每個 anchor 的 triplet 數: {self.triplet_per_anchor}")
        logger.info(f"  文字 embedding 維度: {self.embedding_dim}")
        if self.use_clip and self.clip_processor:
            logger.info(f"  圖片 embedding 維度: {self.clip_embedding_dim} (CLIP)")
        else:
            logger.info(f"  圖片 embedding 維度: {self.embedding_dim} (隨機)")
        
        # 顯示範例
        if len(df) > 0:
            example = df.iloc[0]
            logger.info(f"  範例 triplet:")
            logger.info(f"    Anchor: {example['anchor_text'][:50]}...")
            logger.info(f"    Positive: {example['positive_text'][:50]}...")
            logger.info(f"    Negative: {example['negative_text'][:50]}...")

def main():
    parser = argparse.ArgumentParser(description='Triplet 訓練樣本生成工具')
    parser.add_argument('--samples_file', type=str, required=True, help='樣本檔案路徑')
    parser.add_argument('--leaf_file', type=str, required=True, help='LEAF CSV 檔案路徑')
    parser.add_argument('--root_file', type=str, required=True, help='ROOT CSV 檔案路徑')
    parser.add_argument('--output', type=str, default='triplet_dataset.csv', help='輸出檔案路徑')
    parser.add_argument('--embedding_dim', type=int, default=512, help='文字 Embedding 維度')
    parser.add_argument('--max_text_length', type=int, default=100, help='文字最大長度')
    parser.add_argument('--triplet_per_anchor', type=int, default=3, help='每個 anchor 的 triplet 數量')
    parser.add_argument('--max_triplets', type=int, default=10000, help='最大 triplet 數量')
    parser.add_argument('--use_clip', action='store_true', default=True, help='使用 CLIP 模型')
    parser.add_argument('--no_clip', dest='use_clip', action='store_false', help='不使用 CLIP 模型')
    
    args = parser.parse_args()
    
    generator = TripletDatasetGenerator(
        embedding_dim=args.embedding_dim,
        max_text_length=args.max_text_length,
        triplet_per_anchor=args.triplet_per_anchor,
        use_clip=args.use_clip
    )
    
    generator.generate_triplet_dataset(
        args.samples_file,
        args.leaf_file,
        args.root_file,
        args.output,
        args.max_triplets
    )

if __name__ == "__main__":
    main()
