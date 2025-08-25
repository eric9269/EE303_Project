import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pickle


# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassificationDataset(Dataset):
    """分類資料集"""
    
    def __init__(self, data_path: str, text_embedding_dim: int = 512, image_embedding_dim: int = 512):
        """
        初始化分類資料集
        
        Args:
            data_path: CSV 檔案路徑
            text_embedding_dim: 文字 embedding 維度
            image_embedding_dim: 圖片 embedding 維度
        """
        self.data = pd.read_csv(data_path)
        self.text_embedding_dim = text_embedding_dim
        self.image_embedding_dim = image_embedding_dim
        
        # 轉換 embedding 欄位
        self._process_embeddings()
        
    def _process_embeddings(self):
        """處理 embedding 欄位"""
        embedding_columns = [
            'text_embedding_a', 'text_embedding_b',
            'image_embedding_a', 'image_embedding_b'
        ]
        
        for col in embedding_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(eval)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 文字 embedding
        text_a = torch.tensor(row['text_embedding_a'], dtype=torch.float32)
        text_b = torch.tensor(row['text_embedding_b'], dtype=torch.float32)
        
        # 圖片 embedding
        image_a = torch.tensor(row['image_embedding_a'], dtype=torch.float32)
        image_b = torch.tensor(row['image_embedding_b'], dtype=torch.float32)
        
        # 標籤
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'text_a': text_a,
            'text_b': text_b,
            'image_a': image_a,
            'image_b': image_b,
            'label': label,
            'product_name_a': row['product_name_a'],
            'product_name_b': row['product_name_b']
        }

class ClassificationModel(nn.Module):
    """分類模型"""
    
    def __init__(self, text_embedding_dim: int = 512, image_embedding_dim: int = 512, 
                 hidden_dim: int = 256, num_classes: int = 2):
        super(ClassificationModel, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.image_embedding_dim = image_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 文字處理層
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 圖片處理層
        self.image_encoder = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 特徵融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, text_a, text_b, image_a, image_b):
        """
        前向傳播
        
        Args:
            text_a: 商品 A 文字 embedding
            text_b: 商品 B 文字 embedding
            image_a: 商品 A 圖片 embedding
            image_b: 商品 B 圖片 embedding
            
        Returns:
            分類結果
        """
        # 編碼文字特徵
        text_a_encoded = self.text_encoder(text_a)
        text_b_encoded = self.text_encoder(text_b)
        
        # 編碼圖片特徵
        image_a_encoded = self.image_encoder(image_a)
        image_b_encoded = self.image_encoder(image_b)
        
        # 計算特徵差異
        text_diff = torch.abs(text_a_encoded - text_b_encoded)
        image_diff = torch.abs(image_a_encoded - image_b_encoded)
        
        # 融合特徵
        combined = torch.cat([text_diff, image_diff], dim=1)
        
        # 分類
        output = self.fusion_layer(combined)
        
        return output

class ClassificationTrainer:
    """分類模型訓練器"""
    
    def __init__(self, model: ClassificationModel, device: str = "auto"):
        """
        初始化訓練器
        
        Args:
            model: 分類模型
            device: 設備
        """
        self.model = model
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"使用設備: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        訓練一個 epoch
        
        Args:
            dataloader: 資料載入器
            
        Returns:
            (平均損失, 準確率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            # 移動資料到設備
            text_a = batch['text_a'].to(self.device)
            text_b = batch['text_b'].to(self.device)
            image_a = batch['image_a'].to(self.device)
            image_b = batch['image_b'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向傳播
            outputs = self.model(text_a, text_b, image_a, image_b)
            
            # 計算損失
            loss = self.criterion(outputs, labels)
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 計算準確率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """
        驗證模型
        
        Args:
            dataloader: 驗證資料載入器
            
        Returns:
            (平均損失, 準確率, 詳細指標)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 移動資料到設備
                text_a = batch['text_a'].to(self.device)
                text_b = batch['text_b'].to(self.device)
                image_a = batch['image_a'].to(self.device)
                image_b = batch['image_b'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向傳播
                outputs = self.model(text_a, text_b, image_a, image_b)
                
                # 計算損失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 計算準確率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 收集預測結果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        # 計算詳細指標
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return avg_loss, accuracy, metrics
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'text_embedding_dim': self.model.text_embedding_dim,
                'image_embedding_dim': self.model.image_embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes
            }
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"模型已從 {path} 載入")

def train_classification_model(data_path: str, 
                             output_dir: str = "classification_model",
                             text_embedding_dim: int = 512,
                             image_embedding_dim: int = 512,
                             hidden_dim: int = 256,
                             batch_size: int = 32,
                             epochs: int = 50,
                             learning_rate: float = 0.001,
                             train_split: float = 0.8,
                             device: str = "auto"):
    """
    訓練分類模型
    
    Args:
        data_path: 訓練資料路徑
        output_dir: 輸出目錄
        text_embedding_dim: 文字 embedding 維度
        image_embedding_dim: 圖片 embedding 維度
        hidden_dim: 隱藏層維度
        batch_size: 批次大小
        epochs: 訓練輪數
        learning_rate: 學習率
        train_split: 訓練集比例
        device: 設備
    """
    logger.info("開始訓練分類模型...")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 載入資料
    dataset = ClassificationDataset(data_path, text_embedding_dim, image_embedding_dim)
    
    # 分割訓練和驗證集
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 創建資料載入器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建模型
    model = ClassificationModel(text_embedding_dim, image_embedding_dim, hidden_dim, num_classes=2)
    
    # 創建訓練器
    trainer = ClassificationTrainer(model, device)
    trainer.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練記錄
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    best_metrics = None
    
    # 訓練循環
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_metrics = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        logger.info(f"  Val   - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_metrics = val_metrics
            trainer.save_model(output_path / "best_classification_model.pth")
        
        # 每 10 個 epoch 保存一次檢查點
        if (epoch + 1) % 10 == 0:
            trainer.save_model(output_path / f"classification_model_epoch_{epoch+1}.pth")
    
    # 保存最終模型
    trainer.save_model(output_path / "final_classification_model.pth")
    
    # 保存訓練歷史
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'best_metrics': best_metrics,
        'config': {
            'text_embedding_dim': text_embedding_dim,
            'image_embedding_dim': image_embedding_dim,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
    }
    
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # 保存分類報告
    if best_metrics:
        report = classification_report(
            best_metrics['labels'], 
            best_metrics['predictions'],
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        with open(output_path / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    logger.info(f"訓練完成！最佳驗證準確率: {best_val_accuracy:.4f}")
    logger.info(f"模型和記錄已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='分類模型訓練')
    parser.add_argument('--data_path', type=str, required=True, help='訓練資料路徑')
    parser.add_argument('--output_dir', type=str, default='classification_model', help='輸出目錄')
    parser.add_argument('--text_embedding_dim', type=int, default=512, help='文字 embedding 維度')
    parser.add_argument('--image_embedding_dim', type=int, default=512, help='圖片 embedding 維度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隱藏層維度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--train_split', type=float, default=0.8, help='訓練集比例')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    
    args = parser.parse_args()
    
    train_classification_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        text_embedding_dim=args.text_embedding_dim,
        image_embedding_dim=args.image_embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        device=args.device
    )

if __name__ == "__main__":
    main()
