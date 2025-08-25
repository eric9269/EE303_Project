"""
工具驗證腳本
驗證所有 Python 工具是否都能正常工作
"""

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolValidator:
    """工具驗證器"""
    
    def __init__(self, tools_dir: str = "."):
        """
        初始化驗證器
        
        Args:
            tools_dir: tools 目錄路徑
        """
        self.tools_dir = Path(tools_dir)
        self.results = {}
        
    def validate_imports(self, file_path: Path) -> Tuple[bool, str]:
        """
        驗證 Python 檔案的 import 是否正常
        
        Args:
            file_path: Python 檔案路徑
            
        Returns:
            (是否成功, 錯誤訊息)
        """
        try:
            # 獲取模組名稱
            module_name = file_path.stem
            
            # 添加目錄到 Python 路徑
            sys.path.insert(0, str(file_path.parent))
            
            # 嘗試導入模組
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                return False, "無法創建模組規格"
                
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                return False, "無法獲取模組載入器"
                
            spec.loader.exec_module(module)
            
            # 移除添加的路徑
            sys.path.pop(0)
            
            return True, "導入成功"
            
        except Exception as e:
            # 移除添加的路徑
            if str(file_path.parent) in sys.path:
                sys.path.remove(str(file_path.parent))
            return False, f"導入失敗: {str(e)}"
    
    def validate_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """
        驗證 Python 檔案的語法是否正確
        
        Args:
            file_path: Python 檔案路徑
            
        Returns:
            (是否成功, 錯誤訊息)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 編譯檢查語法
            compile(content, file_path.name, 'exec')
            return True, "語法正確"
            
        except SyntaxError as e:
            return False, f"語法錯誤: {str(e)}"
        except Exception as e:
            return False, f"讀取檔案失敗: {str(e)}"
    
    def validate_help(self, file_path: Path) -> Tuple[bool, str]:
        """
        驗證 Python 檔案的 help 功能是否正常
        
        Args:
            file_path: Python 檔案路徑
            
        Returns:
            (是否成功, 錯誤訊息)
        """
        try:
            # 檢查是否有 main 函數或 argparse
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 檢查是否有 argparse 或 main 函數
            has_argparse = 'argparse' in content
            has_main = 'if __name__ == "__main__"' in content
            
            if not has_main:
                return True, "無 main 函數，跳過 help 測試"
            
            if not has_argparse:
                return True, "無 argparse，跳過 help 測試"
            
            # 嘗試執行 help
            result = subprocess.run(
                [sys.executable, str(file_path), '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, "help 功能正常"
            else:
                return False, f"help 執行失敗: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "help 執行超時"
        except Exception as e:
            return False, f"help 測試失敗: {str(e)}"
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        驗證單個 Python 檔案
        
        Args:
            file_path: Python 檔案路徑
            
        Returns:
            驗證結果
        """
        logger.info(f"驗證檔案: {file_path}")
        
        result = {
            'file': str(file_path),
            'syntax': False,
            'import': False,
            'help': False,
            'errors': []
        }
        
        # 驗證語法
        syntax_ok, syntax_msg = self.validate_syntax(file_path)
        result['syntax'] = syntax_ok
        if not syntax_ok:
            result['errors'].append(f"語法錯誤: {syntax_msg}")
        
        # 驗證導入
        import_ok, import_msg = self.validate_imports(file_path)
        result['import'] = import_ok
        if not import_ok:
            result['errors'].append(f"導入錯誤: {import_msg}")
        
        # 驗證 help
        help_ok, help_msg = self.validate_help(file_path)
        result['help'] = help_ok
        if not help_ok:
            result['errors'].append(f"Help 錯誤: {help_msg}")
        
        return result
    
    def find_python_files(self) -> List[Path]:
        """
        查找所有 Python 檔案
        
        Returns:
            Python 檔案列表
        """
        python_files = []
        
        for root, dirs, files in os.walk(self.tools_dir):
            # 跳過 __pycache__ 目錄
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def validate_all(self) -> Dict[str, Any]:
        """
        驗證所有 Python 檔案
        
        Returns:
            驗證結果
        """
        logger.info("開始驗證所有 Python 檔案...")
        
        python_files = self.find_python_files()
        logger.info(f"找到 {len(python_files)} 個 Python 檔案")
        
        results = []
        total_files = len(python_files)
        passed_files = 0
        
        for file_path in python_files:
            result = self.validate_file(file_path)
            results.append(result)
            
            # 檢查是否通過所有測試
            if result['syntax'] and result['import']:
                passed_files += 1
        
        # 統計結果
        summary = {
            'total_files': total_files,
            'passed_files': passed_files,
            'failed_files': total_files - passed_files,
            'pass_rate': passed_files / total_files if total_files > 0 else 0,
            'results': results
        }
        
        self.results = summary
        return summary
    
    def print_results(self):
        """打印驗證結果"""
        if not self.results:
            logger.error("沒有驗證結果")
            return
        
        print("\n" + "="*60)
        print("工具驗證結果")
        print("="*60)
        
        summary = self.results
        print(f"總檔案數: {summary['total_files']}")
        print(f"通過檔案數: {summary['passed_files']}")
        print(f"失敗檔案數: {summary['failed_files']}")
        print(f"通過率: {summary['pass_rate']:.2%}")
        
        print("\n詳細結果:")
        print("-"*60)
        
        for result in summary['results']:
            status = "✅" if result['syntax'] and result['import'] else "❌"
            print(f"{status} {result['file']}")
            
            if result['errors']:
                for error in result['errors']:
                    print(f"    └─ {error}")
        
        print("\n" + "="*60)
    
    def save_results(self, output_file: str = "validation_results.json"):
        """保存驗證結果到檔案"""
        if not self.results:
            logger.error("沒有驗證結果")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"驗證結果已保存到: {output_file}")

def validate_specific_tools():
    """驗證特定工具的依賴"""
    logger.info("檢查特定工具的依賴...")
    
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
            importlib.import_module(module)
            available_deps.append(name)
            logger.info(f"✅ {name} 可用")
        except ImportError:
            missing_deps.append(name)
            logger.warning(f"❌ {name} 不可用")
    
    print("\n" + "="*60)
    print("依賴檢查結果")
    print("="*60)
    print(f"可用的依賴: {', '.join(available_deps)}")
    if missing_deps:
        print(f"缺失的依賴: {', '.join(missing_deps)}")
        print("\n安裝缺失依賴:")
        print("pip install torch transformers clip-by-openai scikit-learn pandas numpy requests pillow")
    else:
        print("所有依賴都已安裝！")
    print("="*60)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='驗證所有 Python 工具')
    parser.add_argument('--tools_dir', type=str, default='.', help='tools 目錄路徑')
    parser.add_argument('--output', type=str, default='validation_results.json', help='結果輸出檔案')
    parser.add_argument('--check_deps', action='store_true', help='檢查依賴')
    
    args = parser.parse_args()
    
    # 檢查依賴
    if args.check_deps:
        validate_specific_tools()
    
    # 驗證工具
    validator = ToolValidator(args.tools_dir)
    results = validator.validate_all()
    
    # 打印結果
    validator.print_results()
    
    # 保存結果
    validator.save_results(args.output)
    
    # 返回狀態碼
    if results['failed_files'] > 0:
        logger.warning(f"有 {results['failed_files']} 個檔案驗證失敗")
        sys.exit(1)
    else:
        logger.info("所有檔案驗證通過！")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    import importlib.util
    main()
