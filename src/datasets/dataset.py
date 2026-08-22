# File: src/datasets/dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as transforms

# Tránh lỗi khi đọc phải file ảnh bị hỏng/cắt xén trong tập dữ liệu lớn
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Tham số:
        - csv_file: Đường dẫn tới file csv nhãn (VD: nih_labels.csv)
        - image_dir: Đường dẫn tới thư mục chứa ảnh.
        - transform: Các phép biến đổi ảnh (Data Augmentation)
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Tự động nhận diện cấu trúc CSV (Dùng được cho cả NIH và VinBigData)
        # Cột 0: ID ảnh | Cột 1: Đường dẫn ảnh cũ | Cột 2 trở đi: 14 nhãn bệnh lý
        self.label_cols = self.df.columns[2:].tolist()
        self.num_classes = len(self.label_cols)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Bóc tách tên file ảnh bất chấp đường dẫn gốc trong CSV là gì
        filename = os.path.basename(row['image_path'])
        
        # Ghép với thư mục chứa ảnh hiện tại (Tương thích PC / Laptop / Kaggle)
        img_path = os.path.join(self.image_dir, filename)
        
        # Nếu mất file ảnh, báo lỗi rõ ràng thay vì văng app
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ LỖI: Không tìm thấy ảnh tại {img_path}. Hãy kiểm tra lại thư mục!")
            
        # Đọc ảnh và chuyển sang RGB (Mạng Deep Learning cần 3 kênh màu)
        image = Image.open(img_path).convert('RGB')
        
        # Áp dụng các phép biến đổi ảnh
        if self.transform:
            image = self.transform(image)
            
        # Ép kiểu nhãn về float32 (Bắt buộc đối với hàm BCE Loss trong Pytorch)
        labels = row[self.label_cols].values.astype('float32')
        labels = torch.tensor(labels)
        
        return image, labels


def get_dataloaders(csv_file, image_dir, batch_size=32, is_train=True, num_workers=0):
    """
    Khởi tạo DataLoader với Data Augmentation.
    Lưu ý: Trên PC Windows, num_workers nên để = 0 khi test để tránh lỗi đa luồng (Broken Pipe).
    Khi mang lên Kaggle train thật, sẽ đổi num_workers = 2 hoặc 4 sau.
    """
    # Chuẩn hóa theo ImageNet (Vì dùng mô hình Pre-trained DenseNet121)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if is_train:
        # Chế độ Train: Thêm lật ảnh và xoay ảnh để chống Overfitting
        transform = transforms.Compose([
            transforms.Resize((256, 256)),      # Thu nhỏ ảnh về 256x256 để PC chạy nhẹ nhàng
            transforms.RandomHorizontalFlip(),  # Lật ngang ảnh ngẫu nhiên
            transforms.RandomRotation(10),      # Xoay ngẫu nhiên +- 10 độ
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Chế độ Test/Val: Tuyệt đối không Augmented (Không lật, không xoay)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize
        ])
        
    dataset = ChestXrayDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers, 
        pin_memory=True          # Giúp đẩy dữ liệu vào RAM/VRAM nhanh hơn
    )
    
    return dataloader, dataset.label_cols