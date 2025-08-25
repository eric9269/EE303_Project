#!/usr/bin/env python3
"""
簡單但正確的 MySQL 檔案解析器
基於真實的欄位結構分析
"""

import pandas as pd
from pathlib import Path
import re
import json

def parse_real_structure():
    """解析真實的欄位結構"""
    print("解析真實的 MySQL 表格結構")
    print("=" * 80)
    
    # 基於我們之前的分析，真正的欄位結構應該是：
    # 1. product_id - 產品識別碼
    # 2. image_urls - 圖片URL列表
    # 3. product_page_url - 產品頁面URL
    # 4. price_info - 價格資訊
    # 5. shop_info - 商店資訊
    # 6. category_info - 分類資訊
    # 7. metadata - 其他元資料
    
    # 從我們之前提取的資料中重新組織
    data_dir = Path("../src/product_matching/extracted_data")
    
    if not data_dir.exists():
        print("❌ extracted_data 目錄不存在")
        print("請確保 src/product_matching/extracted_data 目錄存在")
        return
    
    # 選擇代表性的表格
    representative_files = [
        ("test_data.csv", "Test Data"),
        ("pm_leaf_v1_extracted.csv", "LEAF v1"),
        ("pm_root_v1_extracted.csv", "ROOT v1"),
        ("pm_leaf_it_shopee_v1_extracted.csv", "LEAF IT Shopee"),
        ("pm_root_it_shopee_v1_extracted.csv", "ROOT IT Shopee")
    ]
    
    all_results = {}
    
    for file_name, table_name in representative_files:
        file_path = data_dir / file_name
        
        if file_path.exists():
            print(f"\n📄 重新組織 {file_name}")
            print("-" * 40)
            
            # 重新組織資料
            reorganized_data = reorganize_data(str(file_path), table_name)
            
            if reorganized_data:
                all_results[table_name] = reorganized_data
    
    # 保存重新組織的資料
    save_reorganized_data(all_results)
    
    return all_results

def reorganize_data(file_path, table_name):
    """重新組織資料為正確的欄位結構"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        print(f"原始資料: {len(df.columns)} 欄位 x {len(df)} 行")
        
        # 重新組織資料
        reorganized_rows = []
        
        for i in range(min(50, len(df))):  # 只處理前50行
            row_data = df.iloc[i]
            
            # 收集所有非空資料
            all_data = []
            for col_idx, value in enumerate(row_data):
                if pd.notna(value) and str(value).strip():
                    all_data.append(str(value))
            
            # 分析資料結構
            row_analysis = analyze_row_data(all_data, i)
            
            if row_analysis:
                reorganized_rows.append(row_analysis)
        
        print(f"重新組織: {len(reorganized_rows)} 行")
        
        # 統計分析
        if reorganized_rows:
            total_product_ids = sum(len(row['product_ids']) for row in reorganized_rows)
            total_image_urls = sum(len(row['image_urls']) for row in reorganized_rows)
            total_page_urls = sum(len(row['page_urls']) for row in reorganized_rows)
            total_prices = sum(len(row['prices']) for row in reorganized_rows)
            
            print(f"  產品ID: {total_product_ids} 個")
            print(f"  圖片URL: {total_image_urls} 個")
            print(f"  頁面URL: {total_page_urls} 個")
            print(f"  價格: {total_prices} 個")
            
            avg_image_urls = total_image_urls / len(reorganized_rows)
            print(f"  平均每行圖片URL: {avg_image_urls:.1f}")
        
        return reorganized_rows
        
    except Exception as e:
        print(f"❌ 重新組織 {file_path} 時發生錯誤: {e}")
        return None

def analyze_row_data(all_data, row_index):
    """分析單行資料"""
    try:
        # 提取 URL
        urls = []
        for data_piece in all_data:
            urls.extend(re.findall(r'https?://[^\s\x00]+', data_piece))
        
        # 提取產品 ID
        product_ids = []
        for data_piece in all_data:
            product_ids.extend(re.findall(r'\d{10,}', data_piece))
        
        # 提取價格
        prices = []
        for data_piece in all_data:
            prices.extend(re.findall(r'\d+\s*-\s*\d+', data_piece))
        
        # 提取商店名稱
        shops = []
        shop_patterns = [
            r'([a-zA-Z0-9_]+)ruten',
            r'([a-zA-Z0-9_]+)shopee',
            r'([a-zA-Z0-9_]+)momo'
        ]
        for data_piece in all_data:
            for pattern in shop_patterns:
                shops.extend(re.findall(pattern, data_piece))
        
        # 分類 URL
        image_urls = [url for url in urls if 'rimg.com.tw' in url or 'img' in url]
        page_urls = [url for url in urls if 'goods.ruten.com.tw' in url or 'item/show' in url]
        
        # 去重
        product_ids = list(set(product_ids))
        image_urls = list(set(image_urls))
        page_urls = list(set(page_urls))
        prices = list(set(prices))
        shops = list(set(shops))
        
        # 構建結構化記錄
        record = {
            'row_index': row_index,
            'product_ids': product_ids,
            'image_urls': image_urls,
            'page_urls': page_urls,
            'prices': prices,
            'shops': shops,
            'other_data': extract_other_data(all_data)
        }
        
        return record
        
    except Exception as e:
        print(f"分析第 {row_index} 行時發生錯誤: {e}")
        return None

def extract_other_data(all_data):
    """提取其他資料"""
    try:
        # 合併所有資料
        combined = ' '.join(all_data)
        
        # 移除已知的資料類型
        cleaned = combined
        
        # 移除 URL
        cleaned = re.sub(r'https?://[^\s\x00]+', '', cleaned)
        
        # 移除產品 ID
        cleaned = re.sub(r'\d{10,}', '', cleaned)
        
        # 移除價格
        cleaned = re.sub(r'\d+\s*-\s*\d+', '', cleaned)
        
        # 移除商店名稱
        cleaned = re.sub(r'[a-zA-Z0-9_]+(ruten|shopee|momo)', '', cleaned)
        
        # 清理空白字元
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 只保留有意义的資料
        if len(cleaned) > 10:
            return cleaned[:200]  # 限制長度
        return ""
        
    except Exception as e:
        return ""

def save_reorganized_data(all_results):
    """保存重新組織的資料"""
    print(f"\n💾 保存重新組織的資料")
    print("=" * 60)
    
    output_dir = Path("../data/correct_structure_data")
    output_dir.mkdir(exist_ok=True)
    
    for table_name, records in all_results.items():
        if not records:
            continue
            
        # 轉換為 DataFrame
        df_data = []
        
        for record in records:
            row = {
                'product_ids': json.dumps(record['product_ids'], ensure_ascii=False),
                'image_urls': json.dumps(record['image_urls'], ensure_ascii=False),
                'page_urls': json.dumps(record['page_urls'], ensure_ascii=False),
                'prices': json.dumps(record['prices'], ensure_ascii=False),
                'shops': json.dumps(record['shops'], ensure_ascii=False),
                'link': record['other_data']
            }
            df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # 保存為 CSV
            output_file = output_dir / f"{table_name.lower().replace(' ', '_')}_correct.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✅ 已保存: {output_file}")
            print(f"   記錄數: {len(df)}")
            print(f"   欄位數: {len(df.columns)}")
            print(f"   欄位: {list(df.columns)}")
            
            # 顯示範例
            if len(df) > 0:
                print(f"   範例產品ID: {df.iloc[0]['product_ids'][:100]}...")
                print(f"   範例圖片URL數量: {len(json.loads(df.iloc[0]['image_urls']))}")

def main():
    """主函數"""
    results = parse_real_structure()
    
    if results:
        print(f"\n📊 重新組織完成總結:")
        print("=" * 60)
        
        for table_name, records in results.items():
            if records:
                print(f"\n{table_name}:")
                print(f"  記錄數: {len(records)}")
                
                avg_product_ids = sum(len(r['product_ids']) for r in records) / len(records)
                avg_image_urls = sum(len(r['image_urls']) for r in records) / len(records)
                avg_page_urls = sum(len(r['page_urls']) for r in records) / len(records)
                
                print(f"  平均產品ID數: {avg_product_ids:.1f}")
                print(f"  平均圖片URL數: {avg_image_urls:.1f}")
                print(f"  平均頁面URL數: {avg_page_urls:.1f}")
                
                print(f"  真實欄位結構:")
                print(f"    1. product_ids - 產品識別碼列表")
                print(f"    2. image_urls - 圖片URL列表 ({avg_image_urls:.0f} 個)")
                print(f"    3. page_urls - 產品頁面URL列表")
                print(f"    4. prices - 價格資訊列表")
                print(f"    5. shops - 商店資訊列表")
                print(f"    6. link - 對應的ROOT ID")

if __name__ == "__main__":
    main()
