#!/usr/bin/env python3
"""
查看正確結構的資料
"""

import pandas as pd
from pathlib import Path
import json

def view_correct_data():
    """查看正確結構的資料"""
    print("查看正確結構的 MySQL 表格資料")
    print("=" * 80)
    
    data_dir = Path("../data/correct_structure_data")
    
    if not data_dir.exists():
        print("❌ correct_structure_data 目錄不存在")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ 沒有找到 CSV 檔案")
        return
    
    print(f"✅ 找到 {len(csv_files)} 個正確結構的檔案:")
    for i, file_path in enumerate(csv_files, 1):
        size_kb = file_path.stat().st_size / 1024
        print(f"{i:2d}. {file_path.name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 80)
    print("正確結構資料預覽:")
    print("=" * 80)
    
    for file_path in csv_files:
        print(f"\n📋 {file_path.name}")
        print("-" * 60)
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            print(f"表格大小: {len(df.columns)} 欄位 x {len(df)} 行")
            print(f"欄位名稱: {list(df.columns)}")
            
            print(f"\n前3行資料預覽:")
            for i in range(min(3, len(df))):
                print(f"\n第 {i+1} 行:")
                row = df.iloc[i]
                
                # 解析 JSON 欄位
                try:
                    # 處理可能的 NaN 值
                    if pd.isna(row['product_ids']):
                        product_ids = []
                    else:
                        product_ids = json.loads(row['product_ids'])
                    
                    if pd.isna(row['image_urls']):
                        image_urls = []
                    else:
                        image_urls = json.loads(row['image_urls'])
                    
                    if pd.isna(row['page_urls']):
                        page_urls = []
                    else:
                        page_urls = json.loads(row['page_urls'])
                    
                    if pd.isna(row['prices']):
                        prices = []
                    else:
                        prices = json.loads(row['prices'])
                    
                    if pd.isna(row['shops']):
                        shops = []
                    else:
                        shops = json.loads(row['shops'])
                    
                    print(f"  產品ID: {len(product_ids)} 個 {product_ids[:3]}")
                    print(f"  圖片URL: {len(image_urls)} 個")
                    if image_urls:
                        print(f"    範例: {image_urls[0][:80]}...")
                    print(f"  頁面URL: {len(page_urls)} 個")
                    if page_urls:
                        print(f"    範例: {page_urls[0][:80]}...")
                    print(f"  價格: {prices}")
                    print(f"  商店: {shops}")
                    print(f"  對應ROOT ID: {row['link'][:100] if pd.notna(row['link']) else ''}...")
                    
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"  資料格式錯誤，無法解析 JSON: {e}")
            
            # 統計分析
            print(f"\n📊 統計分析:")
            total_product_ids = 0
            total_image_urls = 0
            total_page_urls = 0
            
            for i in range(len(df)):
                try:
                    # 處理可能的 NaN 值
                    if pd.isna(df.iloc[i]['product_ids']):
                        product_ids = []
                    else:
                        product_ids = json.loads(df.iloc[i]['product_ids'])
                    
                    if pd.isna(df.iloc[i]['image_urls']):
                        image_urls = []
                    else:
                        image_urls = json.loads(df.iloc[i]['image_urls'])
                    
                    if pd.isna(df.iloc[i]['page_urls']):
                        page_urls = []
                    else:
                        page_urls = json.loads(df.iloc[i]['page_urls'])
                    
                    total_product_ids += len(product_ids)
                    total_image_urls += len(image_urls)
                    total_page_urls += len(page_urls)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            print(f"  總產品ID數: {total_product_ids}")
            print(f"  總圖片URL數: {total_image_urls}")
            print(f"  總頁面URL數: {total_page_urls}")
            
            if len(df) > 0:
                print(f"  平均每行產品ID: {total_product_ids/len(df):.1f}")
                print(f"  平均每行圖片URL: {total_image_urls/len(df):.1f}")
                print(f"  平均每行頁面URL: {total_page_urls/len(df):.1f}")
            
        except Exception as e:
            print(f"❌ 讀取 {file_path} 時發生錯誤: {e}")

def main():
    """主函數"""
    view_correct_data()

if __name__ == "__main__":
    main()
