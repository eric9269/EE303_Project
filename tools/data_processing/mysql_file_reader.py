#!/usr/bin/env python3
"""
MySQL 檔案直接讀取工具
作為無法連接 MySQL 服務時的備選方案
直接讀取 MySQL 資料檔案
"""

import os
import struct
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MySQLFileReader:
    """MySQL 檔案直接讀取器"""
    
    def __init__(self, data_dir: str = "../../xampp/mysql/data/product_matching"):
        """
        初始化檔案讀取器
        
        Args:
            data_dir: MySQL 資料目錄路徑
        """
        self.data_dir = data_dir
        self.tables = {}
        
    def scan_tables(self) -> List[str]:
        """
        掃描資料目錄中的表格
        
        Returns:
            List[str]: 表格名稱列表
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"資料目錄不存在: {self.data_dir}")
            return []
        
        tables = set()
        
        # 掃描 .frm 檔案來獲取表格名稱
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.frm'):
                table_name = filename.replace('.frm', '')
                tables.add(table_name)
        
        return sorted(list(tables))
    
    def get_table_files(self, table_name: str) -> Dict[str, str]:
        """
        獲取表格相關的檔案
        
        Args:
            table_name: 表格名稱
            
        Returns:
            Dict[str, str]: 檔案類型到檔案路徑的映射
        """
        files = {}
        
        # 檢查 .frm 檔案（表格結構）
        frm_file = os.path.join(self.data_dir, f"{table_name}.frm")
        if os.path.exists(frm_file):
            files['frm'] = frm_file
        
        # 檢查 .ibd 檔案（InnoDB 資料）
        ibd_file = os.path.join(self.data_dir, f"{table_name}.ibd")
        if os.path.exists(ibd_file):
            files['ibd'] = ibd_file
        
        return files
    
    def read_frm_structure(self, frm_file: str) -> Dict[str, Any]:
        """
        讀取 .frm 檔案結構（簡化版本）
        
        Args:
            frm_file: .frm 檔案路徑
            
        Returns:
            Dict[str, Any]: 表格結構資訊
        """
        try:
            with open(frm_file, 'rb') as f:
                # 讀取檔案頭部資訊
                header = f.read(64)
                
                # 解析基本資訊
                info = {
                    'file_size': os.path.getsize(frm_file),
                    'table_name': os.path.basename(frm_file).replace('.frm', ''),
                    'has_data': os.path.exists(frm_file.replace('.frm', '.ibd'))
                }
                
                return info
                
        except Exception as e:
            logger.error(f"讀取 .frm 檔案時發生錯誤: {e}")
            return {}
    
    def read_ibd_data(self, ibd_file: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        讀取 .ibd 檔案資料（簡化版本）
        注意：這是簡化的實現，實際的 InnoDB 檔案格式很複雜
        
        Args:
            ibd_file: .ibd 檔案路徑
            limit: 限制讀取的行數
            
        Returns:
            pd.DataFrame: 資料（如果無法解析則返回空 DataFrame）
        """
        try:
            file_size = os.path.getsize(ibd_file)
            logger.info(f"檔案大小: {file_size} bytes")
            
            # 由於 InnoDB 檔案格式複雜，這裡只提供基本資訊
            # 實際的資料解析需要更複雜的實現
            
            with open(ibd_file, 'rb') as f:
                # 讀取檔案頭部
                header = f.read(16384)  # 讀取前 16KB
                
                # 檢查檔案魔術數字
                magic = header[0:4]
                if magic != b'\x00\x00\x00\x00':
                    logger.warning("可能不是有效的 InnoDB 檔案")
                
                # 返回基本檔案資訊
                info_data = {
                    'file_path': ibd_file,
                    'file_size_mb': round(file_size / (1024 * 1024), 2),
                    'file_type': 'InnoDB Data File',
                    'note': '實際資料需要 MySQL 服務來解析'
                }
                
                return pd.DataFrame([info_data])
                
        except Exception as e:
            logger.error(f"讀取 .ibd 檔案時發生錯誤: {e}")
            return pd.DataFrame()
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        獲取表格資訊
        
        Args:
            table_name: 表格名稱
            
        Returns:
            Dict[str, Any]: 表格資訊
        """
        files = self.get_table_files(table_name)
        
        info = {
            'table_name': table_name,
            'files': files,
            'has_structure': 'frm' in files,
            'has_data': 'ibd' in files
        }
        
        if 'frm' in files:
            frm_info = self.read_frm_structure(files['frm'])
            info.update(frm_info)
        
        if 'ibd' in files:
            info['data_file_size_mb'] = round(os.path.getsize(files['ibd']) / (1024 * 1024), 2)
        
        return info
    
    def export_table_info(self, output_file: str = "table_info.csv"):
        """
        匯出所有表格資訊
        
        Args:
            output_file: 輸出檔案名稱
        """
        tables = self.scan_tables()
        
        if not tables:
            logger.warning("沒有找到任何表格")
            return
        
        table_info_list = []
        
        for table in tables:
            info = self.get_table_info(table)
            table_info_list.append(info)
        
        # 轉換為 DataFrame
        df = pd.DataFrame(table_info_list)
        
        # 匯出為 CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"表格資訊已匯出到: {output_file}")
        
        return df


def main():
    """主函數 - 示範如何使用 MySQLFileReader"""
    
    print("MySQL 檔案直接讀取工具")
    print("=" * 50)
    
    # 創建讀取器
    reader = MySQLFileReader()
    
    # 掃描表格
    tables = reader.scan_tables()
    
    if not tables:
        print("❌ 沒有找到任何表格檔案")
        print("請確認資料目錄路徑是否正確")
        return
    
    print(f"✅ 找到 {len(tables)} 個表格:")
    for table in tables:
        print(f"  - {table}")
    
    print("\n" + "=" * 50)
    print("表格詳細資訊:")
    print("=" * 50)
    
    # 顯示每個表格的詳細資訊
    for table in tables:
        print(f"\n表格: {table}")
        info = reader.get_table_info(table)
        
        print(f"  結構檔案: {'✅' if info['has_structure'] else '❌'}")
        print(f"  資料檔案: {'✅' if info['has_data'] else '❌'}")
        
        if info['has_data']:
            print(f"  資料檔案大小: {info.get('data_file_size_mb', 0)} MB")
    
    # 匯出表格資訊
    print(f"\n匯出表格資訊...")
    df = reader.export_table_info()
    
    if df is not None:
        print("\n表格資訊摘要:")
        print(df[['table_name', 'has_structure', 'has_data', 'data_file_size_mb']].to_string(index=False))
    
    print(f"\n注意事項:")
    print(f"1. 這些是 MySQL 的原始資料檔案")
    print(f"2. 要讀取實際資料，需要啟動 MySQL 服務")
    print(f"3. 或者使用 MySQL 工具匯出資料")


if __name__ == "__main__":
    main()
