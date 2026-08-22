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
from utils.metrics import compute_metrics, print_metrics_to_console

from losses.standard_bce import StandardBCE
from losses.focal_loss import FocalLoss
from losses.asymmetric_loss import AsymmetricLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="🚀 Training")
    
    for batch_idx, (images, labels) in prog_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Dùng AMP (Mixed Precision) giúp tăng tốc và giảm một nửa VRAM
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        prog_bar.set_postfix({'loss': f"{running_loss / (batch_idx + 1):.4f}"})
        
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_targets, all_outputs = [], []
    
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="🔎 Validating")
    
    for batch_idx, (images, labels) in prog_bar:
        images, labels = images.to(device), labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        running_loss += loss.item()
        all_targets.append(labels.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())
        
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    val_loss = running_loss / len(dataloader)
    metrics_dict = compute_metrics(all_targets, all_outputs, class_names)
    
    return val_loss, metrics_dict


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 THIẾT BỊ HUẤN LUYỆN: {device.type.upper()} | DATASET: {args.dataset_name.upper()}")
    
    # Chuẩn bị dữ liệu
    df_full = pd.read_csv(args.csv_file)
    train_df, val_df = train_test_split(df_full, test_size=0.2, random_state=42)
    
    os.makedirs('data/splits', exist_ok=True)
    train_csv, val_csv = 'data/splits/temp_train.csv', 'data/splits/temp_val.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"⏳ Nạp Dataloader (Batch Size: {args.batch_size})...")
    train_loader, class_names = get_dataloaders(train_csv, args.img_dir, args.batch_size, is_train=True, num_workers=args.num_workers)
    val_loader, _ = get_dataloaders(val_csv, args.img_dir, args.batch_size, is_train=False, num_workers=args.num_workers)
    
    # Khởi tạo Mô hình & Bộ tối ưu
    model = build_model(num_classes=14, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    # Chọn loss
    if args.loss_type == 'focal':
        print("🎯 SỬ DỤNG HÀM: FOCAL LOSS (Giải quyết Long-tailed)")
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
    elif args.loss_type == 'asl':
        print("🎯 SỬ DỤNG HÀM: ASYMMETRIC LOSS (Giải quyết Đa nhãn)")
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        
    else:
        print("🎯 SỬ DỤNG HÀM: STANDARD BCE (Baseline Đối chứng)")
        criterion = StandardBCE()
        
    best_macro_auc = 0.0
    
    # Vòng lặp Train
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*20} EPOCH {epoch}/{args.epochs} {'='*20}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, class_names)
        
        print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print_metrics_to_console(metrics)
        
        # Lưu Model với Tên linh hoạt
        macro_auc = metrics['Macro_Average']['AUC']
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            os.makedirs('experiments/weights', exist_ok=True)
            
            # Tự động xuất tên: VD 'best_nih_asl_model.pth'
            save_path = f"experiments/weights/best_{args.dataset_name}_{args.loss_type}_model.pth"
            
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Đã lưu mô hình tốt nhất vào {save_path} (Macro-AUC: {best_macro_auc:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Dataset định danh (nih hoặc vin)
    parser.add_argument('--dataset_name', type=str, choices=['nih', 'vin'], default='nih')
    
    # Đường dẫn trỏ thẳng vào bộ dữ liệu chuẩn trên PC/Laptop của em
    parser.add_argument('--csv_file', type=str, default="data/processed/nih_labels.csv")
    parser.add_argument('--img_dir', type=str, default="data/processed/nih_512x512/images")
    
    # Cấu hình an toàn để test thử (2 epoch, batch 8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0) 
    
    # Công tắc đổi Hàm Loss
    parser.add_argument('--loss_type', type=str, choices=['bce', 'focal', 'asl'], default='bce')
    
    args = parser.parse_args()
    main(args)