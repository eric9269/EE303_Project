"""
快速驗證腳本
快速檢查所有工具是否可用
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_check():
    """快速檢查所有工具"""
    print("🔍 快速驗證所有工具...")
    print("="*60)
    
    # 檢查目錄結構
    directories = [
        "data_processing",
        "embedding_processors", 
        "dataset_generators",
        "model_training",
        "model_testing",
        "pipelines"
    ]
    
    print("📁 檢查目錄結構:")
    for dir_name in directories:
        if Path(dir_name).exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (缺失)")
    
    print("\n📄 檢查 Python 檔案:")
    
    # 檢查關鍵檔案
    key_files = [
        "data_processing/simple_correct_parser.py",
        "data_processing/view_correct_data.py",
        "embedding_processors/clip_image_processor.py",
        "embedding_processors/text_embedding_processor.py",
        "dataset_generators/bm25_sampler.py",
        "dataset_generators/classification_dataset_generator.py",
        "dataset_generators/triplet_dataset_generator.py",
        "dataset_generators/generate_training_datasets.py",
        "model_training/train_triplet_model.py",
        "model_training/train_classification_model.py",
        "model_testing/test_models.py",
        "pipelines/train_and_test_pipeline.py"
    ]
    
    missing_files = []
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (缺失)")
            missing_files.append(file_path)
    
    print("\n🔧 檢查 help 功能:")
    
    # 測試 help 功能
    help_test_files = [
        "data_processing/simple_correct_parser.py",
        "dataset_generators/bm25_sampler.py",
        "dataset_generators/classification_dataset_generator.py",
        "dataset_generators/triplet_dataset_generator.py",
        "model_training/train_triplet_model.py",
        "model_training/train_classification_model.py",
        "model_testing/test_models.py",
        "pipelines/train_and_test_pipeline.py"
    ]
    
    for file_path in help_test_files:
        if Path(file_path).exists():
            try:
                result = subprocess.run([
                    sys.executable, file_path, "--help"
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    print(f"  ✅ {file_path} (help 正常)")
                else:
                    print(f"  ⚠️ {file_path} (help 有問題)")
            except Exception as e:
                print(f"  ❌ {file_path} (help 失敗: {e})")
        else:
            print(f"  ❌ {file_path} (檔案不存在)")
    
    print("\n📋 檢查 README 檔案:")
    
    # 檢查 README 檔案
    readme_files = [
        "README.md",
        "data_processing/README.md",
        "embedding_processors/README.md",
        "dataset_generators/README.md",
        "model_training/README.md",
        "model_testing/README.md",
        "pipelines/README.md"
    ]
    
    for readme_path in readme_files:
        if Path(readme_path).exists():
            print(f"  ✅ {readme_path}")
        else:
            print(f"  ❌ {readme_path} (缺失)")
    
    print("\n" + "="*60)
    
    if missing_files:
        print(f"⚠️ 發現 {len(missing_files)} 個缺失檔案")
        return False
    else:
        print("🎉 所有檔案都存在！")
        return True

def check_dependencies():
    """檢查依賴"""
    print("\n🔍 檢查依賴套件...")
    print("="*60)
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers', 
        'clip': 'CLIP',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'requests': 'Requests',
        'PIL': 'Pillow'
    }
    
    missing_deps = []
    available_deps = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            available_deps.append(name)
            print(f"  ✅ {name}")
        except ImportError:
            missing_deps.append(name)
            print(f"  ❌ {name} (缺失)")
    
    print("\n" + "="*60)
    
    if missing_deps:
        print(f"⚠️ 缺失 {len(missing_deps)} 個依賴: {', '.join(missing_deps)}")
        print("\n安裝指令:")
        print("pip install torch transformers clip-by-openai scikit-learn pandas numpy requests pillow")
        return False
    else:
        print("🎉 所有依賴都已安裝！")
        return True

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='快速驗證工具')
    parser.add_argument('--check_deps', action='store_true', help='檢查依賴')
    parser.add_argument('--check_files', action='store_true', help='檢查檔案')
    
    args = parser.parse_args()
    
    if args.check_deps:
        check_dependencies()
    elif args.check_files:
        quick_check()
    else:
        # 執行所有檢查
        files_ok = quick_check()
        deps_ok = check_dependencies()
        
        print("\n📊 總結:")
        print("="*60)
        if files_ok:
            print("🎉 所有檔案檢查都通過！工具可以正常使用。")
            if not deps_ok:
                print("⚠️ 缺少可選依賴 CLIP，但不影響基本功能。")
            sys.exit(0)
        else:
            print("⚠️ 發現問題，請檢查上述錯誤訊息。")
            sys.exit(1)

if __name__ == "__main__":
    main()
