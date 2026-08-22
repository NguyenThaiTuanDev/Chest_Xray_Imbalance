# File: src/utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import warnings

# Bỏ qua các cảnh báo chia cho 0 của sklearn để log terminal sạch sẽ
warnings.filterwarnings('ignore')

def compute_metrics(y_true, y_pred_logits, class_names, threshold=0.5):
    # Áp dụng hàm Sigmoid để chuyển logits thành xác suất (Probabilities) từ 0 -> 1
    # Công thức Sigmoid: 1 / (1 + exp(-x))
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
    
    # Chuyển xác suất thành nhãn nhị phân (0 hoặc 1) dựa vào threshold để tính F1
    y_pred_bin = (y_pred_probs >= threshold).astype(float)
    
    num_classes = len(class_names)
    aucs = []
    f1s = []
    
    metrics_dict = {}
    
    for i in range(num_classes):
        class_name = class_names[i]
        
        # --- TÍNH ROC-AUC ---
        # Bắt lỗi ValueError: sklearn sẽ báo lỗi nếu trong tập batch đó 
        # không có tấm ảnh nào mang bệnh này (chỉ toàn số 0).
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            auc = np.nan # Gán NaN (Not a Number) nếu không thể tính
            
        aucs.append(auc)
        
        # --- TÍNH F1-SCORE ---
        # zero_division=0 giúp tránh lỗi chia cho 0 nếu mô hình không đoán trúng phát nào
        f1 = f1_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
        f1s.append(f1)
        
        metrics_dict[class_name] = {
            'AUC': auc,
            'F1': f1
        }
        
    # Tính chỉ số trung bình (Macro Average)
    # Bỏ qua các giá trị NaN bằng np.nanmean
    macro_auc = np.nanmean(aucs)
    macro_f1 = np.nanmean(f1s)
    
    metrics_dict['Macro_Average'] = {
        'AUC': macro_auc,
        'F1': macro_f1
    }
    
    return metrics_dict