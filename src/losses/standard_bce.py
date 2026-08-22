import torch.nn as nn

class StandardBCE(nn.Module):
    def __init__(self):
        super(StandardBCE, self).__init__()
        # BCEWithLogitsLoss tự động áp dụng Sigmoid bên trong, 
        # Rất an toàn về mặt toán học và tránh lỗi tràn số (overflow).
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, logits, targets):
        """
        logits: Dự đoán của mô hình (Shape: Batch_size x 14)
        targets: Nhãn One-hot thực tế (Shape: Batch_size x 14)
        """
        loss = self.criterion(logits, targets)
        return loss