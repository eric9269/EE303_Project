import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pickle

# 導入模型類別
from tools.model_training.train_triplet_model import SimilarityModel, TripletDataset
from tools.model_training.train_classification_model import ClassificationModel, ClassificationDataset

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    """模型測試器"""
    
    def __init__(self, device: str = "auto"):
        """
        初始化測試器
        
        Args:
            device: 設備
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"使用設備: {self.device}")
    
    def load_similarity_model(self, model_path: str) -> SimilarityModel:
        """載入相似度模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']
        
        model = SimilarityModel(
            text_embedding_dim=config['text_embedding_dim'],
            image_embedding_dim=config['image_embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"相似度模型已載入: {model_path}")
        return model
    
    def load_classification_model(self, model_path: str) -> ClassificationModel:
        """載入分類模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['model_config']
        
        model = ClassificationModel(
            text_embedding_dim=config['text_embedding_dim'],
            image_embedding_dim=config['image_embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"分類模型已載入: {model_path}")
        return model
    
    def test_similarity_model(self, model: SimilarityModel, test_data_path: str, 
                            batch_size: int = 32) -> Dict[str, Any]:
        """
        測試相似度模型
        
        Args:
            model: 相似度模型
            test_data_path: 測試資料路徑
            batch_size: 批次大小
            
        Returns:
            測試結果
        """
        logger.info("開始測試相似度模型...")
        
        # 載入測試資料
        test_dataset = TripletDataset(test_data_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        total_loss = 0.0
        all_similarities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 移動資料到設備
                anchor_text = batch['anchor_text'].to(self.device)
                positive_text = batch['positive_text'].to(self.device)
                negative_text = batch['negative_text'].to(self.device)
                anchor_image = batch['anchor_image'].to(self.device)
                positive_image = batch['positive_image'].to(self.device)
                negative_image = batch['negative_image'].to(self.device)
                
                # 前向傳播
                anchor_embedding = model(anchor_text, anchor_image)
                positive_embedding = model(positive_text, positive_image)
                negative_embedding = model(negative_text, negative_image)
                
                # 計算相似度
                pos_similarity = torch.cosine_similarity(anchor_embedding, positive_embedding, dim=1)
                neg_similarity = torch.cosine_similarity(anchor_embedding, negative_embedding, dim=1)
                
                all_similarities.extend([
                    {'positive': pos.item(), 'negative': neg.item()} 
                    for pos, neg in zip(pos_similarity, neg_similarity)
                ])
        
        # 計算統計指標
        positive_similarities = [s['positive'] for s in all_similarities]
        negative_similarities = [s['negative'] for s in all_similarities]
        
        avg_pos_sim = np.mean(positive_similarities)
        avg_neg_sim = np.mean(negative_similarities)
        sim_gap = avg_pos_sim - avg_neg_sim
        
        # 計算準確率（正樣本相似度 > 負樣本相似度）
        correct = sum(1 for s in all_similarities if s['positive'] > s['negative'])
        accuracy = correct / len(all_similarities)
        
        results = {
            'avg_positive_similarity': avg_pos_sim,
            'avg_negative_similarity': avg_neg_sim,
            'similarity_gap': sim_gap,
            'accuracy': accuracy,
            'total_samples': len(all_similarities),
            'correct_predictions': correct
        }
        
        logger.info(f"相似度模型測試結果:")
        logger.info(f"  平均正樣本相似度: {avg_pos_sim:.4f}")
        logger.info(f"  平均負樣本相似度: {avg_neg_sim:.4f}")
        logger.info(f"  相似度差距: {sim_gap:.4f}")
        logger.info(f"  準確率: {accuracy:.4f}")
        
        return results
    
    def test_classification_model(self, model: ClassificationModel, test_data_path: str,
                                batch_size: int = 32) -> Dict[str, Any]:
        """
        測試分類模型
        
        Args:
            model: 分類模型
            test_data_path: 測試資料路徑
            batch_size: 批次大小
            
        Returns:
            測試結果
        """
        logger.info("開始測試分類模型...")
        
        # 載入測試資料
        test_dataset = ClassificationDataset(test_data_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 移動資料到設備
                text_a = batch['text_a'].to(self.device)
                text_b = batch['text_b'].to(self.device)
                image_a = batch['image_a'].to(self.device)
                image_b = batch['image_b'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向傳播
                outputs = model(text_a, text_b, image_a, image_b)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 獲取預測結果
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 計算指標
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 生成分類報告
        report = classification_report(
            all_labels, all_predictions,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'total_samples': len(all_labels)
        }
        
        logger.info(f"分類模型測試結果:")
        logger.info(f"  準確率: {accuracy:.4f}")
        logger.info(f"  精確率: {precision:.4f}")
        logger.info(f"  召回率: {recall:.4f}")
        logger.info(f"  F1 分數: {f1:.4f}")
        
        return results
    
    def test_two_stage_model(self, similarity_model: SimilarityModel, 
                           classification_model: ClassificationModel,
                           test_data_path: str, similarity_threshold: float = 0.5,
                           batch_size: int = 32) -> Dict[str, Any]:
        """
        測試兩階段組合模型
        
        Args:
            similarity_model: 相似度模型
            classification_model: 分類模型
            test_data_path: 測試資料路徑
            similarity_threshold: 相似度閾值
            batch_size: 批次大小
            
        Returns:
            測試結果
        """
        logger.info("開始測試兩階段組合模型...")
        
        # 載入測試資料
        test_dataset = ClassificationDataset(test_data_path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        similarity_model.eval()
        classification_model.eval()
        
        all_predictions = []
        all_labels = []
        all_similarities = []
        stage1_passed = 0
        stage2_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 移動資料到設備
                text_a = batch['text_a'].to(self.device)
                text_b = batch['text_b'].to(self.device)
                image_a = batch['image_a'].to(self.device)
                image_b = batch['image_b'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 第一階段：相似度計算
                similarity_embedding_a = similarity_model(text_a, image_a)
                similarity_embedding_b = similarity_model(text_b, image_b)
                similarities = torch.cosine_similarity(similarity_embedding_a, similarity_embedding_b, dim=1)
                
                all_similarities.extend(similarities.cpu().numpy())
                
                # 根據相似度閾值過濾
                passed_mask = similarities >= similarity_threshold
                stage1_passed += passed_mask.sum().item()
                
                # 第二階段：分類
                outputs = classification_model(text_a, text_b, image_a, image_b)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                # 組合預測結果
                final_predictions = torch.zeros_like(predicted)
                final_predictions[passed_mask] = predicted[passed_mask]  # 通過第一階段的用分類結果
                final_predictions[~passed_mask] = 0  # 未通過第一階段的直接判定為負樣本
                
                all_predictions.extend(final_predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 記錄第二階段預測
                stage2_predictions.extend(predicted.cpu().numpy())
        
        # 計算指標
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 計算各階段統計
        avg_similarity = np.mean(all_similarities)
        stage1_pass_rate = stage1_passed / len(all_labels)
        
        # 生成分類報告
        report = classification_report(
            all_labels, all_predictions,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'avg_similarity': avg_similarity,
            'stage1_pass_rate': stage1_pass_rate,
            'similarity_threshold': similarity_threshold,
            'predictions': all_predictions,
            'labels': all_labels,
            'similarities': all_similarities,
            'stage2_predictions': stage2_predictions,
            'total_samples': len(all_labels),
            'stage1_passed': stage1_passed
        }
        
        logger.info(f"兩階段組合模型測試結果:")
        logger.info(f"  相似度閾值: {similarity_threshold}")
        logger.info(f"  平均相似度: {avg_similarity:.4f}")
        logger.info(f"  第一階段通過率: {stage1_pass_rate:.4f}")
        logger.info(f"  最終準確率: {accuracy:.4f}")
        logger.info(f"  最終精確率: {precision:.4f}")
        logger.info(f"  最終召回率: {recall:.4f}")
        logger.info(f"  最終 F1 分數: {f1:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='模型測試')
    parser.add_argument('--test_data_path', type=str, required=True, help='測試資料路徑')
    parser.add_argument('--similarity_model_path', type=str, help='相似度模型路徑')
    parser.add_argument('--classification_model_path', type=str, help='分類模型路徑')
    parser.add_argument('--output_dir', type=str, default='test_results', help='輸出目錄')
    parser.add_argument('--similarity_threshold', type=float, default=0.5, help='相似度閾值')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 創建測試器
    tester = ModelTester(args.device)
    
    results = {}
    
    # 測試相似度模型
    if args.similarity_model_path:
        similarity_model = tester.load_similarity_model(args.similarity_model_path)
        similarity_results = tester.test_similarity_model(
            similarity_model, args.test_data_path, args.batch_size
        )
        results['similarity_model'] = similarity_results
        
        # 保存結果
        with open(output_path / "similarity_test_results.json", 'w') as f:
            json.dump(similarity_results, f, indent=2)
    
    # 測試分類模型
    if args.classification_model_path:
        classification_model = tester.load_classification_model(args.classification_model_path)
        classification_results = tester.test_classification_model(
            classification_model, args.test_data_path, args.batch_size
        )
        results['classification_model'] = classification_results
        
        # 保存結果
        with open(output_path / "classification_test_results.json", 'w') as f:
            json.dump(classification_results, f, indent=2)
    
    # 測試兩階段組合模型
    if args.similarity_model_path and args.classification_model_path:
        two_stage_results = tester.test_two_stage_model(
            similarity_model, classification_model, args.test_data_path,
            args.similarity_threshold, args.batch_size
        )
        results['two_stage_model'] = two_stage_results
        
        # 保存結果
        with open(output_path / "two_stage_test_results.json", 'w') as f:
            json.dump(two_stage_results, f, indent=2)
    
    # 保存總體結果
    with open(output_path / "all_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"所有測試結果已保存到: {output_path}")

if __name__ == "__main__":
    main()
