import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

def build_model(num_classes=14, pretrained=True):
    # Sử dụng bộ trọng số đã huấn luyện trước trên ImageNet (giúp hội tụ cực nhanh)
    if pretrained:
        weights = DenseNet121_Weights.DEFAULT 
    else:
        weights = None
        
    model = densenet121(weights=weights)
    
    # Lấy kích thước đầu ra của lớp đặc trưng cuối cùng (của DenseNet121 là 1024)
    in_features = model.classifier.in_features
    
    # Thay thế lớp phân loại cuối (từ 1000 class của ImageNet thành 14 class của bài toán)
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model