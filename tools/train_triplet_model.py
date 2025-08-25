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
import pickle

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    """Triplet 資料集"""
    
    def __init__(self, data_path: str, text_embedding_dim: int = 512, image_embedding_dim: int = 512):
        """
        初始化 Triplet 資料集
        
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
            'anchor_text_embedding', 'positive_text_embedding', 'negative_text_embedding',
            'anchor_image_embedding', 'positive_image_embedding', 'negative_image_embedding'
        ]
        
        for col in embedding_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(eval)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 文字 embedding
        anchor_text = torch.tensor(row['anchor_text_embedding'], dtype=torch.float32)
        positive_text = torch.tensor(row['positive_text_embedding'], dtype=torch.float32)
        negative_text = torch.tensor(row['negative_text_embedding'], dtype=torch.float32)
        
        # 圖片 embedding
        anchor_image = torch.tensor(row['anchor_image_embedding'], dtype=torch.float32)
        positive_image = torch.tensor(row['positive_image_embedding'], dtype=torch.float32)
        negative_image = torch.tensor(row['negative_image_embedding'], dtype=torch.float32)
        
        return {
            'anchor_text': anchor_text,
            'positive_text': positive_text,
            'negative_text': negative_text,
            'anchor_image': anchor_image,
            'positive_image': positive_image,
            'negative_image': negative_image,
            'anchor_text_raw': row['anchor_text'],
            'positive_text_raw': row['positive_text'],
            'negative_text_raw': row['negative_text']
        }

class TripletLoss(nn.Module):
    """Triplet Loss"""
    
    def __init__(self, margin: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        計算 Triplet Loss
        
        Args:
            anchor: anchor embedding
            positive: positive embedding
            negative: negative embedding
            
        Returns:
            triplet loss
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

class SimilarityModel(nn.Module):
    """相似度模型"""
    
    def __init__(self, text_embedding_dim: int = 512, image_embedding_dim: int = 512, 
                 hidden_dim: int = 256, output_dim: int = 128):
        super(SimilarityModel, self).__init__()
        
        self.text_embedding_dim = text_embedding_dim
        self.image_embedding_dim = image_embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 文字處理層
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.L2Normalization()
        )
        
        # 圖片處理層
        self.image_encoder = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.L2Normalization()
        )
        
        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.L2Normalization()
        )
    
    def encode_text(self, text_embedding):
        """編碼文字"""
        return self.text_encoder(text_embedding)
    
    def encode_image(self, image_embedding):
        """編碼圖片"""
        return self.image_encoder(image_embedding)
    
    def forward(self, text_embedding, image_embedding):
        """
        前向傳播
        
        Args:
            text_embedding: 文字 embedding
            image_embedding: 圖片 embedding
            
        Returns:
            融合後的 embedding
        """
        text_encoded = self.encode_text(text_embedding)
        image_encoded = self.encode_image(image_embedding)
        
        # 融合文字和圖片特徵
        combined = torch.cat([text_encoded, image_encoded], dim=1)
        fused = self.fusion_layer(combined)
        
        return fused

class TripletTrainer:
    """Triplet 模型訓練器"""
    
    def __init__(self, model: SimilarityModel, device: str = "auto"):
        """
        初始化訓練器
        
        Args:
            model: 相似度模型
            device: 設備
        """
        self.model = model
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        self.criterion = TripletLoss(margin=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"使用設備: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        訓練一個 epoch
        
        Args:
            dataloader: 資料載入器
            
        Returns:
            平均損失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # 移動資料到設備
            anchor_text = batch['anchor_text'].to(self.device)
            positive_text = batch['positive_text'].to(self.device)
            negative_text = batch['negative_text'].to(self.device)
            anchor_image = batch['anchor_image'].to(self.device)
            positive_image = batch['positive_image'].to(self.device)
            negative_image = batch['negative_image'].to(self.device)
            
            # 前向傳播
            anchor_embedding = self.model(anchor_text, anchor_image)
            positive_embedding = self.model(positive_text, positive_image)
            negative_embedding = self.model(negative_text, negative_image)
            
            # 計算損失
            loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        驗證模型
        
        Args:
            dataloader: 驗證資料載入器
            
        Returns:
            平均損失
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # 移動資料到設備
                anchor_text = batch['anchor_text'].to(self.device)
                positive_text = batch['positive_text'].to(self.device)
                negative_text = batch['negative_text'].to(self.device)
                anchor_image = batch['anchor_image'].to(self.device)
                positive_image = batch['positive_image'].to(self.device)
                negative_image = batch['negative_image'].to(self.device)
                
                # 前向傳播
                anchor_embedding = self.model(anchor_text, anchor_image)
                positive_embedding = self.model(positive_text, positive_image)
                negative_embedding = self.model(negative_text, negative_image)
                
                # 計算損失
                loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'text_embedding_dim': self.model.text_embedding_dim,
                'image_embedding_dim': self.model.image_embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim
            }
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"模型已從 {path} 載入")

def train_triplet_model(data_path: str, 
                       output_dir: str = "triplet_model",
                       text_embedding_dim: int = 512,
                       image_embedding_dim: int = 512,
                       hidden_dim: int = 256,
                       output_dim: int = 128,
                       batch_size: int = 32,
                       epochs: int = 50,
                       learning_rate: float = 0.001,
                       train_split: float = 0.8,
                       device: str = "auto"):
    """
    訓練 Triplet 模型
    
    Args:
        data_path: 訓練資料路徑
        output_dir: 輸出目錄
        text_embedding_dim: 文字 embedding 維度
        image_embedding_dim: 圖片 embedding 維度
        hidden_dim: 隱藏層維度
        output_dim: 輸出維度
        batch_size: 批次大小
        epochs: 訓練輪數
        learning_rate: 學習率
        train_split: 訓練集比例
        device: 設備
    """
    logger.info("開始訓練 Triplet 模型...")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 載入資料
    dataset = TripletDataset(data_path, text_embedding_dim, image_embedding_dim)
    
    # 分割訓練和驗證集
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 創建資料載入器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 創建模型
    model = SimilarityModel(text_embedding_dim, image_embedding_dim, hidden_dim, output_dim)
    
    # 創建訓練器
    trainer = TripletTrainer(model, device)
    trainer.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練記錄
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 訓練循環
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model(output_path / "best_triplet_model.pth")
        
        # 每 10 個 epoch 保存一次檢查點
        if (epoch + 1) % 10 == 0:
            trainer.save_model(output_path / f"triplet_model_epoch_{epoch+1}.pth")
    
    # 保存最終模型
    trainer.save_model(output_path / "final_triplet_model.pth")
    
    # 保存訓練歷史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': {
            'text_embedding_dim': text_embedding_dim,
            'image_embedding_dim': image_embedding_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
    }
    
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"訓練完成！最佳驗證損失: {best_val_loss:.4f}")
    logger.info(f"模型和記錄已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Triplet 模型訓練')
    parser.add_argument('--data_path', type=str, required=True, help='訓練資料路徑')
    parser.add_argument('--output_dir', type=str, default='triplet_model', help='輸出目錄')
    parser.add_argument('--text_embedding_dim', type=int, default=512, help='文字 embedding 維度')
    parser.add_argument('--image_embedding_dim', type=int, default=512, help='圖片 embedding 維度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隱藏層維度')
    parser.add_argument('--output_dim', type=int, default=128, help='輸出維度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='學習率')
    parser.add_argument('--train_split', type=float, default=0.8, help='訓練集比例')
    parser.add_argument('--device', type=str, default='auto', help='設備')
    
    args = parser.parse_args()
    
    train_triplet_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        text_embedding_dim=args.text_embedding_dim,
        image_embedding_dim=args.image_embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        device=args.device
    )

if __name__ == "__main__":
    main()
