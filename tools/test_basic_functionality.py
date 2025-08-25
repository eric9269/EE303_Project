"""
基本功能測試腳本
測試各個工具的基本功能是否正常
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_processing():
    """測試資料處理工具"""
    logger.info("測試資料處理工具...")
    
    try:
        # 測試 simple_correct_parser.py
        result = subprocess.run([
            sys.executable, "data_processing/simple_correct_parser.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ simple_correct_parser.py 正常")
        else:
            logger.warning(f"⚠️ simple_correct_parser.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ simple_correct_parser.py 測試失敗: {e}")
    
    try:
        # 測試 view_correct_data.py
        result = subprocess.run([
            sys.executable, "data_processing/view_correct_data.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ view_correct_data.py 正常")
        else:
            logger.warning(f"⚠️ view_correct_data.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ view_correct_data.py 測試失敗: {e}")

def test_embedding_processors():
    """測試 embedding 處理器"""
    logger.info("測試 embedding 處理器...")
    
    try:
        # 測試 clip_image_processor.py
        result = subprocess.run([
            sys.executable, "embedding_processors/clip_image_processor.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ clip_image_processor.py 正常")
        else:
            logger.warning(f"⚠️ clip_image_processor.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ clip_image_processor.py 測試失敗: {e}")
    
    try:
        # 測試 text_embedding_processor.py
        result = subprocess.run([
            sys.executable, "embedding_processors/text_embedding_processor.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ text_embedding_processor.py 正常")
        else:
            logger.warning(f"⚠️ text_embedding_processor.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ text_embedding_processor.py 測試失敗: {e}")

def test_dataset_generators():
    """測試資料集生成器"""
    logger.info("測試資料集生成器...")
    
    try:
        # 測試 bm25_sampler.py
        result = subprocess.run([
            sys.executable, "dataset_generators/bm25_sampler.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ bm25_sampler.py 正常")
        else:
            logger.warning(f"⚠️ bm25_sampler.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ bm25_sampler.py 測試失敗: {e}")
    
    try:
        # 測試 classification_dataset_generator.py
        result = subprocess.run([
            sys.executable, "dataset_generators/classification_dataset_generator.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ classification_dataset_generator.py 正常")
        else:
            logger.warning(f"⚠️ classification_dataset_generator.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ classification_dataset_generator.py 測試失敗: {e}")
    
    try:
        # 測試 triplet_dataset_generator.py
        result = subprocess.run([
            sys.executable, "dataset_generators/triplet_dataset_generator.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ triplet_dataset_generator.py 正常")
        else:
            logger.warning(f"⚠️ triplet_dataset_generator.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ triplet_dataset_generator.py 測試失敗: {e}")

def test_model_training():
    """測試模型訓練工具"""
    logger.info("測試模型訓練工具...")
    
    try:
        # 測試 train_triplet_model.py
        result = subprocess.run([
            sys.executable, "model_training/train_triplet_model.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ train_triplet_model.py 正常")
        else:
            logger.warning(f"⚠️ train_triplet_model.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ train_triplet_model.py 測試失敗: {e}")
    
    try:
        # 測試 train_classification_model.py
        result = subprocess.run([
            sys.executable, "model_training/train_classification_model.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ train_classification_model.py 正常")
        else:
            logger.warning(f"⚠️ train_classification_model.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ train_classification_model.py 測試失敗: {e}")

def test_model_testing():
    """測試模型測試工具"""
    logger.info("測試模型測試工具...")
    
    try:
        # 測試 test_models.py
        result = subprocess.run([
            sys.executable, "model_testing/test_models.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ test_models.py 正常")
        else:
            logger.warning(f"⚠️ test_models.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ test_models.py 測試失敗: {e}")

def test_pipelines():
    """測試管道工具"""
    logger.info("測試管道工具...")
    
    try:
        # 測試 train_and_test_pipeline.py
        result = subprocess.run([
            sys.executable, "pipelines/train_and_test_pipeline.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ train_and_test_pipeline.py 正常")
        else:
            logger.warning(f"⚠️ train_and_test_pipeline.py 可能有問題: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ train_and_test_pipeline.py 測試失敗: {e}")

def create_test_data():
    """創建測試資料"""
    logger.info("創建測試資料...")
    
    # 創建測試目錄
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # 創建測試 CSV 檔案
    test_csv = test_dir / "test_leaf.csv"
    test_csv.write_text("""id,name,price,description,images,link
1,測試商品1,100,這是測試商品1,["http://example.com/img1.jpg"],1
2,測試商品2,200,這是測試商品2,["http://example.com/img2.jpg"],2
""")
    
    test_root_csv = test_dir / "test_root.csv"
    test_root_csv.write_text("""id,name,price,description,images
1,測試商品1,100,這是測試商品1,["http://example.com/img1.jpg"]
2,測試商品2,200,這是測試商品2,["http://example.com/img2.jpg"]
""")
    
    logger.info("✅ 測試資料創建完成")

def run_basic_tests():
    """運行基本測試"""
    logger.info("開始運行基本功能測試...")
    
    # 創建測試資料
    create_test_data()
    
    # 測試各個模組
    test_data_processing()
    test_embedding_processors()
    test_dataset_generators()
    test_model_training()
    test_model_testing()
    test_pipelines()
    
    logger.info("基本功能測試完成！")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基本功能測試')
    parser.add_argument('--create_test_data', action='store_true', help='創建測試資料')
    parser.add_argument('--test_all', action='store_true', help='運行所有測試')
    
    args = parser.parse_args()
    
    if args.create_test_data:
        create_test_data()
    elif args.test_all:
        run_basic_tests()
    else:
        run_basic_tests()

if __name__ == "__main__":
    main()
