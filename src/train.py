import torch
import torch.optim as optim
from tqdm import tqdm
import os

from datasets.dataset import get_dataloaders
from models.classifier import build_model
from losses.standard_bce import StandardBCE

def train_one_epoch(model, dataloader, criterion, optimizer, device, is_test_run=False):
    model.train() # Chuyển mô hình sang chế độ huấn luyện
    running_loss = 0.0
    
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for batch_idx, (images, labels) in prog_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()            # Xóa gradient cũ
        outputs = model(images)          # Dự đoán
        loss = criterion(outputs, labels)# Tính sai số
        loss.backward()                  # Lan truyền ngược
        optimizer.step()                 # Cập nhật trọng số
        
        running_loss += loss.item()
        prog_bar.set_postfix({'loss': f"{running_loss / (batch_idx + 1):.4f}"})
        
        # ⚠️ CHẾ ĐỘ TEST PC: Chỉ chạy 2 batch rồi ngắt để không đơ CPU
        if is_test_run and batch_idx == 1:
            print("\n[Sanity Check] Đã chạy xong 2 batch đầu tiên. Luồng dữ liệu hoàn hảo!")
            break

def main():
    # 1. Tự động nhận diện thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang chạy thử nghiệm trên: {device.type.upper()}")
    
    # 2. Cấu hình đường dẫn dữ liệu NIH
    CSV_FILE = 'data/processed/nih_labels.csv' 
    IMAGE_DIR = 'data/processed/nih_512x512/images'
    
    if not os.path.exists(CSV_FILE):
        print(f"Lỗi: Không tìm thấy file {CSV_FILE}.")
        return

    # 3. Nạp dữ liệu
    print("⏳ Đang nạp DataLoader...")
    train_loader, label_cols = get_dataloaders(
        csv_file=CSV_FILE, 
        image_dir=IMAGE_DIR, 
        batch_size=32, 
        is_train=True,
        num_workers=0 # Giữ 0 trên PC Windows
    )
    
    # 4. Tải Model & Cấu hình AI
    print("⏳ Đang khởi tạo DenseNet121...")
    model = build_model(num_classes=14, pretrained=True)
    model = model.to(device)
    
    criterion = StandardBCE()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # AdamW tốt hơn Adam thường
    
    # 5. Chạy thử nghiệm
    print("🔥 Bắt đầu Dry Run...")
    train_one_epoch(
        model=model, 
        dataloader=train_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device,
        is_test_run=True # Quan trọng
    )
    print("✅ XUẤT SẮC! Toàn bộ hệ thống đã thông suốt!")

if __name__ == '__main__':
    main()