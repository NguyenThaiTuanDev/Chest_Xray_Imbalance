import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss cho Multi-label.
        - alpha: Cân bằng trọng số giữa lớp Đa số và Thiểu số.
        - gamma: Hệ số phạt. Gamma càng lớn, mô hình càng bị ép phải học các ca bệnh khó (nhãn hiếm).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Chuyển input thành xác suất
        probs = torch.sigmoid(inputs)
        
        # 2. Tính p_t (Xác suất dự đoán đúng)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # 3. Tính alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 4. Tính BCE loss ban đầu
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 5. Áp dụng công thức Focal Loss: FL = alpha_t * (1 - pt)^gamma * BCE
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss