"""
整合的訓練集生成腳本
自動生成 BM25 樣本、分類訓練集和 Triplet 訓練集
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd: str, description: str) -> bool:
    """
    執行命令
    
    Args:
        cmd: 命令字串
        description: 命令描述
        
    Returns:
        是否成功
    """
    logger.info(f"執行: {description}")
    logger.info(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"成功: {description}")
        if result.stdout:
            logger.info(f"輸出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"失敗: {description}")
        logger.error(f"錯誤: {e.stderr}")
        return False

def generate_training_datasets(leaf_file: str,
                              root_file: str,
                              output_dir: str = "training_data",
                              k: int = 5,
                              sample_size: int = 1000,
                              embedding_dim: int = 512,
                              max_text_length: int = 100,
                              triplet_per_anchor: int = 3,
                              max_triplets: int = 10000,
                              use_clip: bool = True,
                              clip_model: str = "clip-vit-large-patch14",
                              use_clip: bool = True) -> bool:
    """
    生成完整的訓練資料集
    
    Args:
        leaf_file: LEAF CSV 檔案路徑
        root_file: ROOT CSV 檔案路徑
        output_dir: 輸出目錄
        k: 每個 leaf 選擇的負樣本數量
        sample_size: 總樣本數量限制
        embedding_dim: Embedding 維度
        max_text_length: 文字最大長度
        triplet_per_anchor: 每個 anchor 的 triplet 數量
        max_triplets: 最大 triplet 數量
        
    Returns:
        是否成功
    """
    logger.info("開始生成完整的訓練資料集...")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 步驟 1: 生成 BM25 樣本
    logger.info("=" * 60)
    logger.info("步驟 1: 生成 BM25 樣本")
    logger.info("=" * 60)
    
    samples_file = output_path / "bm25_samples.json"
    cmd1 = f"python bm25_sampler.py --leaf_file {leaf_file} --root_file {root_file} --k {k} --sample_size {sample_size} --output {samples_file}"
    
    if not run_command(cmd1, "生成 BM25 樣本"):
        return False
    
    # 步驟 2: 生成分類訓練集
    logger.info("=" * 60)
    logger.info("步驟 2: 生成分類訓練集")
    logger.info("=" * 60)
    
    classification_file = output_path / "classification_dataset.csv"
    clip_flag = "--use_clip" if use_clip else "--no_clip"
    cmd2 = f"python classification_dataset_generator.py --samples_file {samples_file} --leaf_file {leaf_file} --root_file {root_file} --output {classification_file} --embedding_dim {embedding_dim} --max_text_length {max_text_length} {clip_flag} --clip_model {clip_model}"
    
    if not run_command(cmd2, "生成分類訓練集"):
        return False
    
    # 步驟 3: 生成 Triplet 訓練集
    logger.info("=" * 60)
    logger.info("步驟 3: 生成 Triplet 訓練集")
    logger.info("=" * 60)
    
    triplet_file = output_path / "triplet_dataset.csv"
    cmd3 = f"python triplet_dataset_generator.py --samples_file {samples_file} --leaf_file {leaf_file} --root_file {root_file} --output {triplet_file} --embedding_dim {embedding_dim} --max_text_length {max_text_length} --triplet_per_anchor {triplet_per_anchor} --max_triplets {max_triplets} {clip_flag} --clip_model {clip_model}"
    
    if not run_command(cmd3, "生成 Triplet 訓練集"):
        return False
    
    # 生成報告
    logger.info("=" * 60)
    logger.info("生成完成報告")
    logger.info("=" * 60)
    
    generate_report(output_path, leaf_file, root_file, k, sample_size, embedding_dim, triplet_per_anchor, max_triplets, use_clip)
    
    return True

def generate_report(output_path: Path,
                   leaf_file: str,
                   root_file: str,
                   k: int,
                   sample_size: int,
                   embedding_dim: int,
                   triplet_per_anchor: int,
                   max_triplets: int,
                   use_clip: bool) -> None:
    """
    生成報告
    
    Args:
        output_path: 輸出目錄
        leaf_file: LEAF 檔案路徑
        root_file: ROOT 檔案路徑
        k: 負樣本數量
        sample_size: 樣本大小
        embedding_dim: Embedding 維度
        triplet_per_anchor: 每個 anchor 的 triplet 數量
        max_triplets: 最大 triplet 數量
    """
    report_file = output_path / "generation_report.md"
    
    report_content = f"""# 訓練資料集生成報告

## 生成時間
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 輸入檔案
- LEAF 檔案: {leaf_file}
- ROOT 檔案: {root_file}

## 參數設定
- k (負樣本數量): {k}
- 樣本大小限制: {sample_size}
- Embedding 維度: {embedding_dim}
- 每個 anchor 的 triplet 數量: {triplet_per_anchor}
- 最大 triplet 數量: {max_triplets}

## 生成檔案

### 1. BM25 樣本
- 檔案: `bm25_samples.json`
- 描述: 包含正負樣本對的 JSON 檔案

### 2. 分類訓練集
- 檔案: `classification_dataset.csv`
- 描述: 用於分類任務的訓練資料
- 格式: [label, product_name_a, product_name_b, text_embedding_a, text_embedding_b, image_embedding_a, image_embedding_b]

### 3. Triplet 訓練集
- 檔案: `triplet_dataset.csv`
- 描述: 用於相似度學習的 triplet 資料
- 格式: [anchor_text, positive_text, negative_text, anchor_text_embedding, positive_text_embedding, negative_text_embedding, anchor_image_embedding, positive_image_embedding, negative_image_embedding]

### 4. 向量化器
- 檔案: `*_vectorizer.pkl`
- 描述: 用於文字向量化的 TF-IDF 向量化器

## 使用方式

### 載入分類訓練集
```python
import pandas as pd
import pickle

# 載入資料
df = pd.read_csv('classification_dataset.csv')

# 載入向量化器
with open('classification_dataset_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
```

### 載入 Triplet 訓練集
```python
import pandas as pd
import pickle

# 載入資料
df = pd.read_csv('triplet_dataset.csv')

# 載入向量化器
with open('triplet_dataset_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
```

## 注意事項
1. 圖片 embedding 目前使用隨機向量模擬，實際使用時需要替換為真實的圖片 embedding 模型
2. 文字 embedding 使用 TF-IDF 向量化，可根據需要替換為其他方法
3. 所有 embedding 都已正規化
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"報告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='整合的訓練資料集生成工具')
    parser.add_argument('--leaf_file', type=str, required=True, help='LEAF CSV 檔案路徑')
    parser.add_argument('--root_file', type=str, required=True, help='ROOT CSV 檔案路徑')
    parser.add_argument('--output_dir', type=str, default='training_data', help='輸出目錄')
    parser.add_argument('--k', type=int, default=5, help='每個 leaf 選擇的負樣本數量')
    parser.add_argument('--sample_size', type=int, default=1000, help='總樣本數量限制')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding 維度')
    parser.add_argument('--max_text_length', type=int, default=100, help='文字最大長度')
    parser.add_argument('--triplet_per_anchor', type=int, default=3, help='每個 anchor 的 triplet 數量')
    parser.add_argument('--max_triplets', type=int, default=10000, help='最大 triplet 數量')
    parser.add_argument('--use_clip', action='store_true', default=True, help='使用 CLIP 模型')
    parser.add_argument('--no_clip', dest='use_clip', action='store_false', help='不使用 CLIP 模型')
    parser.add_argument('--clip_model', type=str, default='clip-vit-large-patch14', help='CLIP 模型名稱')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案是否存在
    if not os.path.exists(args.leaf_file):
        logger.error(f"LEAF 檔案不存在: {args.leaf_file}")
        sys.exit(1)
    
    if not os.path.exists(args.root_file):
        logger.error(f"ROOT 檔案不存在: {args.root_file}")
        sys.exit(1)
    
    # 生成訓練資料集
    success = generate_training_datasets(
        leaf_file=args.leaf_file,
        root_file=args.root_file,
        output_dir=args.output_dir,
        k=args.k,
        sample_size=args.sample_size,
        embedding_dim=args.embedding_dim,
        max_text_length=args.max_text_length,
        triplet_per_anchor=args.triplet_per_anchor,
        max_triplets=args.max_triplets,
        use_clip=args.use_clip,
        clip_model=args.clip_model
    )
    
    if success:
        logger.info("所有訓練資料集生成完成！")
        sys.exit(0)
    else:
        logger.error("訓練資料集生成失敗！")
        sys.exit(1)

if __name__ == "__main__":
    import pandas as pd
    main()
