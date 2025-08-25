#!/usr/bin/env python3
"""
測試 EVA-CLIP 功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.embedding_processors.clip_image_processor import CLIPImageProcessor
from PIL import Image
import numpy as np

def test_clip_models():
    """測試不同的 CLIP 模型"""
    print("🔍 測試 CLIP 模型...")
    
    # 測試圖片
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 測試標準 CLIP
    print("\n📷 測試標準 CLIP 模型...")
    try:
        processor = CLIPImageProcessor("clip-vit-large-patch14")
        embedding = processor.get_image_embedding(test_image)
        print(f"✅ 標準 CLIP 成功 - embedding 形狀: {embedding.shape}")
        print(f"   範例值: {embedding[:5]}")
    except Exception as e:
        print(f"❌ 標準 CLIP 失敗: {e}")
    
    # 測試 EVA-CLIP (會自動回退到標準 CLIP)
    print("\n🚀 測試 EVA-CLIP 模型...")
    try:
        processor = CLIPImageProcessor("EVA02-CLIP-bigE-14-plus_s9B")
        embedding = processor.get_image_embedding(test_image)
        print(f"✅ EVA-CLIP 成功 - embedding 形狀: {embedding.shape}")
        print(f"   範例值: {embedding[:5]}")
    except Exception as e:
        print(f"❌ EVA-CLIP 失敗: {e}")
    
    # 測試文字 embedding
    print("\n📝 測試文字 embedding...")
    try:
        processor = CLIPImageProcessor("clip-vit-large-patch14")
        text_embedding = processor.get_text_embedding("a red car")
        print(f"✅ 文字 embedding 成功 - embedding 形狀: {text_embedding.shape}")
        print(f"   範例值: {text_embedding[:5]}")
    except Exception as e:
        print(f"❌ 文字 embedding 失敗: {e}")
    
    # 測試相似度計算
    print("\n🔗 測試相似度計算...")
    try:
        processor = CLIPImageProcessor("clip-vit-large-patch14")
        similarity = processor.get_similarity(test_image, "a red car")
        print(f"✅ 相似度計算成功 - 分數: {similarity:.4f}")
    except Exception as e:
        print(f"❌ 相似度計算失敗: {e}")

def test_batch_processing():
    """測試批次處理"""
    print("\n🔄 測試批次處理...")
    
    try:
        processor = CLIPImageProcessor("clip-vit-large-patch14")
        
        # 創建多張測試圖片
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue'),
            Image.new('RGB', (224, 224), color='green')
        ]
        
        # 批次處理
        embeddings = processor.batch_process_images(images)
        print(f"✅ 批次處理成功 - embeddings 形狀: {embeddings.shape}")
        print(f"   範例值: {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"❌ 批次處理失敗: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("EVA-CLIP 功能測試")
    print("=" * 60)
    
    test_clip_models()
    test_batch_processing()
    
    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)
