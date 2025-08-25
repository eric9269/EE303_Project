"""
整合的訓練和測試管道
自動執行完整的兩階段模型訓練和測試流程
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import subprocess
import json


# 設置日誌
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

def train_and_test_pipeline(training_data_dir: str,
                           output_dir: str = "pipeline_results",
                           text_embedding_dim: int = 512,
                           image_embedding_dim: int = 512,
                           hidden_dim: int = 256,
                           output_dim: int = 128,
                           batch_size: int = 32,
                           epochs: int = 50,
                           learning_rate: float = 0.001,
                           similarity_threshold: float = 0.5,
                           device: str = "auto"):
    """
    執行完整的訓練和測試管道
    
    Args:
        training_data_dir: 訓練資料目錄
        output_dir: 輸出目錄
        text_embedding_dim: 文字 embedding 維度
        image_embedding_dim: 圖片 embedding 維度
        hidden_dim: 隱藏層維度
        output_dim: 輸出維度
        batch_size: 批次大小
        epochs: 訓練輪數
        learning_rate: 學習率
        similarity_threshold: 相似度閾值
        device: 設備
    """
    logger.info("開始執行完整的訓練和測試管道...")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 檢查訓練資料
    triplet_data = Path(training_data_dir) / "triplet_dataset.csv"
    classification_data = Path(training_data_dir) / "classification_dataset.csv"
    
    if not triplet_data.exists():
        logger.error(f"Triplet 訓練資料不存在: {triplet_data}")
        return False
    
    if not classification_data.exists():
        logger.error(f"Classification 訓練資料不存在: {classification_data}")
        return False
    
    # 步驟 1: 訓練 Triplet 模型
    logger.info("=" * 60)
    logger.info("步驟 1: 訓練 Triplet 模型")
    logger.info("=" * 60)
    
    triplet_model_dir = output_path / "triplet_model"
    cmd1 = f"python train_triplet_model.py --data_path {triplet_data} --output_dir {triplet_model_dir} --text_embedding_dim {text_embedding_dim} --image_embedding_dim {image_embedding_dim} --hidden_dim {hidden_dim} --output_dim {output_dim} --batch_size {batch_size} --epochs {epochs} --learning_rate {learning_rate} --device {device}"
    
    if not run_command(cmd1, "訓練 Triplet 模型"):
        return False
    
    # 步驟 2: 訓練 Classification 模型
    logger.info("=" * 60)
    logger.info("步驟 2: 訓練 Classification 模型")
    logger.info("=" * 60)
    
    classification_model_dir = output_path / "classification_model"
    cmd2 = f"python train_classification_model.py --data_path {classification_data} --output_dir {classification_model_dir} --text_embedding_dim {text_embedding_dim} --image_embedding_dim {image_embedding_dim} --hidden_dim {hidden_dim} --batch_size {batch_size} --epochs {epochs} --learning_rate {learning_rate} --device {device}"
    
    if not run_command(cmd2, "訓練 Classification 模型"):
        return False
    
    # 步驟 3: 測試單個模型
    logger.info("=" * 60)
    logger.info("步驟 3: 測試單個模型")
    logger.info("=" * 60)
    
    test_results_dir = output_path / "test_results"
    best_triplet_model = triplet_model_dir / "best_triplet_model.pth"
    best_classification_model = classification_model_dir / "best_classification_model.pth"
    
    cmd3 = f"python test_models.py --test_data_path {classification_data} --similarity_model_path {best_triplet_model} --classification_model_path {best_classification_model} --output_dir {test_results_dir} --similarity_threshold {similarity_threshold} --batch_size {batch_size} --device {device}"
    
    if not run_command(cmd3, "測試模型"):
        return False
    
    # 步驟 4: 生成報告
    logger.info("=" * 60)
    logger.info("步驟 4: 生成最終報告")
    logger.info("=" * 60)
    
    generate_final_report(output_path, training_data_dir, text_embedding_dim, 
                         image_embedding_dim, hidden_dim, output_dim, batch_size, 
                         epochs, learning_rate, similarity_threshold)
    
    logger.info("=" * 60)
    logger.info("管道執行完成！")
    logger.info(f"所有結果已保存到: {output_path}")
    logger.info("=" * 60)
    
    return True

def generate_final_report(output_path: Path, training_data_dir: str,
                         text_embedding_dim: int, image_embedding_dim: int,
                         hidden_dim: int, output_dim: int, batch_size: int,
                         epochs: int, learning_rate: float, similarity_threshold: float):
    """
    生成最終報告
    
    Args:
        output_path: 輸出目錄
        training_data_dir: 訓練資料目錄
        text_embedding_dim: 文字 embedding 維度
        image_embedding_dim: 圖片 embedding 維度
        hidden_dim: 隱藏層維度
        output_dim: 輸出維度
        batch_size: 批次大小
        epochs: 訓練輪數
        learning_rate: 學習率
        similarity_threshold: 相似度閾值
    """
    report_file = output_path / "pipeline_report.md"
    
    # 讀取測試結果
    test_results_file = output_path / "test_results" / "all_test_results.json"
    test_results = {}
    if test_results_file.exists():
        with open(test_results_file, 'r', encoding='utf-8') as f:
            test_results = json.load(f)
    
    report_content = f"""# 兩階段商品匹配模型訓練和測試報告

## 管道概述

本報告記錄了完整的兩階段商品匹配模型訓練和測試流程。

## 配置參數

- **文字 Embedding 維度**: {text_embedding_dim}
- **圖片 Embedding 維度**: {image_embedding_dim}
- **隱藏層維度**: {hidden_dim}
- **輸出維度**: {output_dim}
- **批次大小**: {batch_size}
- **訓練輪數**: {epochs}
- **學習率**: {learning_rate}
- **相似度閾值**: {similarity_threshold}

## 模型架構

### 第一階段：相似度模型 (Triplet)
- **目的**: 學習商品間的相似度表示
- **輸入**: 商品文字和圖片 embedding
- **輸出**: 128 維相似度 embedding
- **損失函數**: Triplet Loss
- **模型路徑**: `triplet_model/best_triplet_model.pth`

### 第二階段：分類模型 (Classification)
- **目的**: 判斷兩個商品是否匹配
- **輸入**: 兩個商品的文字和圖片 embedding
- **輸出**: 二分類結果 (0: 不匹配, 1: 匹配)
- **損失函數**: Cross Entropy Loss
- **模型路徑**: `classification_model/best_classification_model.pth`

## 訓練資料

- **Triplet 資料**: `{training_data_dir}/triplet_dataset.csv`
- **Classification 資料**: `{training_data_dir}/classification_dataset.csv`

## 測試結果

"""

    # 添加測試結果
    if test_results:
        if 'similarity_model' in test_results:
            sim_results = test_results['similarity_model']
            report_content += f"""
### 相似度模型測試結果

- **平均正樣本相似度**: {sim_results.get('avg_positive_similarity', 0):.4f}
- **平均負樣本相似度**: {sim_results.get('avg_negative_similarity', 0):.4f}
- **相似度差距**: {sim_results.get('similarity_gap', 0):.4f}
- **準確率**: {sim_results.get('accuracy', 0):.4f}
- **總樣本數**: {sim_results.get('total_samples', 0)}
"""
        
        if 'classification_model' in test_results:
            cls_results = test_results['classification_model']
            report_content += f"""
### 分類模型測試結果

- **準確率**: {cls_results.get('accuracy', 0):.4f}
- **精確率**: {cls_results.get('precision', 0):.4f}
- **召回率**: {cls_results.get('recall', 0):.4f}
- **F1 分數**: {cls_results.get('f1', 0):.4f}
- **總樣本數**: {cls_results.get('total_samples', 0)}
"""
        
        if 'two_stage_model' in test_results:
            two_stage_results = test_results['two_stage_model']
            report_content += f"""
### 兩階段組合模型測試結果

- **相似度閾值**: {two_stage_results.get('similarity_threshold', 0):.2f}
- **平均相似度**: {two_stage_results.get('avg_similarity', 0):.4f}
- **第一階段通過率**: {two_stage_results.get('stage1_pass_rate', 0):.4f}
- **最終準確率**: {two_stage_results.get('accuracy', 0):.4f}
- **最終精確率**: {two_stage_results.get('precision', 0):.4f}
- **最終召回率**: {two_stage_results.get('recall', 0):.4f}
- **最終 F1 分數**: {two_stage_results.get('f1', 0):.4f}
- **總樣本數**: {two_stage_results.get('total_samples', 0)}
- **第一階段通過數**: {two_stage_results.get('stage1_passed', 0)}
"""

    report_content += f"""
## 檔案結構

```
{output_path}/
├── triplet_model/              # Triplet 模型訓練結果
│   ├── best_triplet_model.pth
│   ├── final_triplet_model.pth
│   ├── training_history.json
│   └── ...
├── classification_model/       # Classification 模型訓練結果
│   ├── best_classification_model.pth
│   ├── final_classification_model.pth
│   ├── training_history.json
│   ├── classification_report.json
│   └── ...
├── test_results/              # 測試結果
│   ├── all_test_results.json
│   ├── similarity_test_results.json
│   ├── classification_test_results.json
│   ├── two_stage_test_results.json
│   └── ...
└── pipeline_report.md         # 本報告
```

## 使用方式

### 載入模型進行推理

```python
from test_models import ModelTester

# 創建測試器
tester = ModelTester()

# 載入模型
similarity_model = tester.load_similarity_model('triplet_model/best_triplet_model.pth')
classification_model = tester.load_classification_model('classification_model/best_classification_model.pth')

# 進行預測
# ... 實現具體的推理邏輯
```

### 調整參數

可以通過修改以下參數來優化模型性能：

1. **相似度閾值**: 調整 `similarity_threshold` 來平衡精確率和召回率
2. **模型架構**: 修改 `hidden_dim` 和 `output_dim`
3. **訓練參數**: 調整 `batch_size`、`epochs`、`learning_rate`

## 注意事項

1. **文字 Embedding**: 目前使用 TF-IDF，可以替換為 BERT 或其他預訓練模型
2. **圖片 Embedding**: 目前使用 CLIP，可以替換為其他視覺模型
3. **資料品質**: 確保訓練資料的品質和平衡性
4. **硬體需求**: 建議使用 GPU 進行訓練

## 未來改進

1. **模型優化**: 嘗試不同的模型架構和損失函數
2. **資料增強**: 使用資料增強技術提高模型泛化能力
3. **集成學習**: 結合多個模型的預測結果
4. **線上學習**: 實現增量學習以適應新資料
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"最終報告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='整合的訓練和測試管道')
    parser.add_argument('--training_data_dir', type=str, required=True, help='訓練資料目錄')
    parser.add_argument('--output_dir', type=str, default='pipeline_results', help='輸出目錄')
    parser.add_argument('--text_embedding_dim', type=int, default=512, help='文字 embedding 維度')
    parser.add_argument('--image_embedding_dim', type=int, default=512, help='圖片 embedding 維度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隱藏層維度')
    parser.add_argument('--output_dim', type=int, default=128, help='輸出維度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='相似度閾值')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    
    args = parser.parse_args()
    
    # 檢查訓練資料目錄
    if not os.path.exists(args.training_data_dir):
        logger.error(f"訓練資料目錄不存在: {args.training_data_dir}")
        sys.exit(1)
    
    # 執行管道
    success = train_and_test_pipeline(
        training_data_dir=args.training_data_dir,
        output_dir=args.output_dir,
        text_embedding_dim=args.text_embedding_dim,
        image_embedding_dim=args.image_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )
    
    if success:
        logger.info("管道執行成功！")
        sys.exit(0)
    else:
        logger.error("管道執行失敗！")
        sys.exit(1)

if __name__ == "__main__":
    main()
