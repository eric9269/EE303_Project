"""
BM25 相似度計算和正負樣本選擇工具
用於找出正樣本（leaf對應的root）和負樣本（最不相近的樣本）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from typing import List, Tuple, Dict, Any
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Sampler:
    def __init__(self, k: int = 5, sample_size: int = 1000):
        """
        初始化 BM25 採樣器
        
        Args:
            k: 每個 leaf 選擇的負樣本數量
            sample_size: 總樣本數量限制
        """
        self.k = k
        self.sample_size = sample_size
        self.leaf_data = None
        self.root_data = None
        self.vectorizer = None
        self.similarity_matrix = None
        
    def load_data(self, leaf_file: str, root_file: str) -> None:
        """
        載入 LEAF 和 ROOT 資料
        
        Args:
            leaf_file: LEAF CSV 檔案路徑
            root_file: ROOT CSV 檔案路徑
        """
        logger.info(f"載入 LEAF 資料: {leaf_file}")
        self.leaf_data = pd.read_csv(leaf_file, encoding='utf-8-sig')
        
        logger.info(f"載入 ROOT 資料: {root_file}")
        self.root_data = pd.read_csv(root_file, encoding='utf-8-sig')
        
        # 解析 JSON 欄位
        self._parse_json_columns()
        
    def _parse_json_columns(self) -> None:
        """解析 JSON 格式的欄位"""
        for df in [self.leaf_data, self.root_data]:
            for col in ['product_ids', 'image_urls', 'page_urls', 'prices', 'shops']:
                if col in df.columns:
                    df[f'{col}_parsed'] = df[col].apply(
                        lambda x: json.loads(x) if pd.notna(x) and x != '[]' else []
                    )
    
    def extract_text_features(self) -> Tuple[List[str], List[str]]:
        """
        提取文字特徵用於相似度計算
        
        Returns:
            leaf_texts: LEAF 的文字特徵
            root_texts: ROOT 的文字特徵
        """
        logger.info("提取文字特徵...")
        
        def combine_features(row):
            """組合多個欄位作為文字特徵"""
            features = []
            
            # 產品ID
            if 'product_ids_parsed' in row and row['product_ids_parsed']:
                features.extend([str(pid) for pid in row['product_ids_parsed']])
            
            # 商店名稱
            if 'shops_parsed' in row and row['shops_parsed']:
                features.extend([str(shop) for shop in row['shops_parsed']])
            
            # 價格
            if 'prices_parsed' in row and row['prices_parsed']:
                features.extend([str(price) for price in row['prices_parsed']])
            
            # 其他資料
            if 'link' in row and pd.notna(row['link']):
                features.append(str(row['link']))
            
            return ' '.join(features)
        
        leaf_texts = [combine_features(row) for _, row in self.leaf_data.iterrows()]
        root_texts = [combine_features(row) for _, row in self.root_data.iterrows()]
        
        return leaf_texts, root_texts
    
    def calculate_similarity(self) -> None:
        """計算 LEAF 和 ROOT 之間的相似度矩陣"""
        logger.info("計算相似度矩陣...")
        
        leaf_texts, root_texts = self.extract_text_features()
        
        # 使用 TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # 合併所有文字進行向量化
        all_texts = leaf_texts + root_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # 分離 LEAF 和 ROOT 的向量
        leaf_vectors = tfidf_matrix[:len(leaf_texts)]
        root_vectors = tfidf_matrix[len(leaf_texts):]
        
        # 計算相似度矩陣
        self.similarity_matrix = cosine_similarity(leaf_vectors, root_vectors)
        
        logger.info(f"相似度矩陣大小: {self.similarity_matrix.shape}")
    
    def find_positive_samples(self) -> List[Tuple[int, int]]:
        """
        找出正樣本（leaf 對應的 root）
        
        Returns:
            positive_pairs: [(leaf_idx, root_idx), ...]
        """
        logger.info("找出正樣本...")
        
        positive_pairs = []
        
        for leaf_idx, leaf_row in self.leaf_data.iterrows():
            if pd.notna(leaf_row['link']) and leaf_row['link'].strip():
                # 根據 link 欄位找到對應的 root
                link_value = str(leaf_row['link']).strip()
                
                # 在 root 資料中尋找匹配的項目
                for root_idx, root_row in self.root_data.iterrows():
                    # 這裡可以根據實際的 link 邏輯進行匹配
                    # 目前使用簡單的相似度匹配
                    if self.similarity_matrix[leaf_idx][root_idx] > 0.1:  # 閾值可調整
                        positive_pairs.append((leaf_idx, root_idx))
                        break
        
        logger.info(f"找到 {len(positive_pairs)} 個正樣本")
        return positive_pairs
    
    def find_negative_samples(self, positive_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        找出負樣本（最不相近的樣本）
        
        Args:
            positive_pairs: 正樣本對
            
        Returns:
            negative_pairs: [(leaf_idx, root_idx), ...]
        """
        logger.info("找出負樣本...")
        
        negative_pairs = []
        positive_leaf_indices = set(pair[0] for pair in positive_pairs)
        
        for leaf_idx in positive_leaf_indices:
            # 獲取該 leaf 與所有 root 的相似度
            similarities = self.similarity_matrix[leaf_idx]
            
            # 找出相似度最低的 k 個 root
            lowest_indices = np.argsort(similarities)[:self.k]
            
            for root_idx in lowest_indices:
                # 確保不是正樣本
                if (leaf_idx, root_idx) not in positive_pairs:
                    negative_pairs.append((leaf_idx, root_idx))
        
        logger.info(f"找到 {len(negative_pairs)} 個負樣本")
        return negative_pairs
    
    def generate_samples(self, leaf_file: str, root_file: str) -> Dict[str, Any]:
        """
        生成正負樣本
        
        Args:
            leaf_file: LEAF CSV 檔案路徑
            root_file: ROOT CSV 檔案路徑
            
        Returns:
            包含正負樣本的字典
        """
        self.load_data(leaf_file, root_file)
        self.calculate_similarity()
        
        positive_pairs = self.find_positive_samples()
        negative_pairs = self.find_negative_samples(positive_pairs)
        
        # 限制樣本數量
        if len(positive_pairs) > self.sample_size // 2:
            positive_pairs = positive_pairs[:self.sample_size // 2]
        
        if len(negative_pairs) > self.sample_size // 2:
            negative_pairs = negative_pairs[:self.sample_size // 2]
        
        return {
            'positive_samples': positive_pairs,
            'negative_samples': negative_pairs,
            'similarity_matrix': self.similarity_matrix,
            'leaf_data': self.leaf_data,
            'root_data': self.root_data
        }

def main():
    parser = argparse.ArgumentParser(description='BM25 相似度計算和正負樣本選擇')
    parser.add_argument('--leaf_file', type=str, required=True, help='LEAF CSV 檔案路徑')
    parser.add_argument('--root_file', type=str, required=True, help='ROOT CSV 檔案路徑')
    parser.add_argument('--k', type=int, default=5, help='每個 leaf 選擇的負樣本數量')
    parser.add_argument('--sample_size', type=int, default=1000, help='總樣本數量限制')
    parser.add_argument('--output', type=str, default='samples.json', help='輸出檔案路徑')
    
    args = parser.parse_args()
    
    sampler = BM25Sampler(k=args.k, sample_size=args.sample_size)
    results = sampler.generate_samples(args.leaf_file, args.root_file)
    
    # 保存結果
    output_data = {
        'k': args.k,
        'sample_size': args.sample_size,
        'positive_samples': [[int(x) for x in pair] for pair in results['positive_samples']],
        'negative_samples': [[int(x) for x in pair] for pair in results['negative_samples']],
        'total_positive': len(results['positive_samples']),
        'total_negative': len(results['negative_samples'])
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"結果已保存到: {args.output}")
    logger.info(f"正樣本數量: {len(results['positive_samples'])}")
    logger.info(f"負樣本數量: {len(results['negative_samples'])}")

if __name__ == "__main__":
    main()
