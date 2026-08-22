import torch
import torch.nn as nn

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        """
        Hàm Asymmetric Loss chuyên trị mất cân bằng Dương tính/Âm tính trong Multi-label.
        
        Tham số:
        - gamma_neg: Hệ số phạt cho nhãn Âm tính (Mặc định để rất cao = 4, ép mô hình loại bỏ hẳn nhiễu âm tính).
        - gamma_pos: Hệ số phạt cho nhãn Dương tính (Mặc định để thấp = 1, giúp mô hình tập trung học đặc trưng bệnh).
        - clip: Ngưỡng loại bỏ (margin). Nếu xác suất âm tính đã rất thấp (dưới clip), ta bỏ qua không tính loss nữa để dồn gradient cho mẫu khó.
        """
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        x: Logits đầu ra từ mô hình
        y: Nhãn thực tế (0 và 1)
        """
        # 1. Tính xác suất bằng Sigmoid
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # 2. Asymmetric Clipping (Cơ chế nới lỏng cho nhãn Âm tính)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 3. Tính toán Cross Entropy cơ bản (thêm eps để tránh log(0))
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # 4. Asymmetric Focusing (Cơ chế phạt bất đối xứng)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)  
        pt = pt0 + pt1
        
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        # 5. Ráp vào công thức ASL
        loss *= one_sided_w

        return -loss.mean()