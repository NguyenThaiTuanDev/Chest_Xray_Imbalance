import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from datasets.dataset import get_dataloaders
from models.classifier import build_model
from losses.standard_bce import StandardBCE
from utils.metrics import compute_metrics, print_metrics_to_console

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="🚀 Training")
    
    for batch_idx, (images, labels) in prog_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        prog_bar.set_postfix({'loss': f"{running_loss / (batch_idx + 1):.4f}"})
        
    return running_loss / len(dataloader)

@torch.no_grad() # Tắt tính toán đạo hàm để tiết kiệm VRAM khi Test
def evaluate(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="🔎 Validating")
    
    for batch_idx, (images, labels) in prog_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # Gom kết quả lại để tính AUC
        all_targets.append(labels.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())
        
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    val_loss = running_loss / len(dataloader)
    metrics_dict = compute_metrics(all_targets, all_outputs, class_names)
    
    return val_loss, metrics_dict

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 BẮT ĐẦU HUẤN LUYỆN TRÊN THIẾT BỊ: {device.type.upper()}")
    
    # 1. Tách tập Train/Val tự động (Tỉ lệ 80/20)
    df_full = pd.read_csv(args.csv_file)
    train_df, val_df = train_test_split(df_full, test_size=0.2, random_state=42)
    
    # Lưu tạm ra file CSV để Dataloader đọc
    train_csv = 'temp_train.csv'
    val_csv = 'temp_val.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # 2. Khởi tạo Dataloaders
    print("⏳ Nạp DataLoader...")
    train_loader, class_names = get_dataloaders(train_csv, args.img_dir, args.batch_size, is_train=True, num_workers=args.num_workers)
    val_loader, _ = get_dataloaders(val_csv, args.img_dir, args.batch_size, is_train=False, num_workers=args.num_workers)
    
    # 3. Khởi tạo Mô hình & Loss
    model = build_model(num_classes=14, pretrained=True).to(device)
    criterion = StandardBCE()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_macro_auc = 0.0
    
    # 4. Vòng lặp Huấn luyện (Epochs)
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*20} EPOCH {epoch}/{args.epochs} {'='*20}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, class_names)
        
        print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print_metrics_to_console(metrics)
        
        macro_auc = metrics['Macro_Average']['AUC']
        
        # 5. Lưu mô hình tốt nhất dựa vào Macro-AUC
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            save_path = f"best_baseline_bce.pth"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Đã lưu mô hình tốt nhất tại Epoch {epoch} với Macro-AUC: {best_macro_auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, required=True, help="Đường dẫn file CSV nhãn gốc")
    parser.add_argument('--img_dir', type=str, required=True, help="Đường dẫn thư mục chứa ảnh")
    parser.add_argument('--epochs', type=int, default=10, help="Số epoch huấn luyện")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    main(args)