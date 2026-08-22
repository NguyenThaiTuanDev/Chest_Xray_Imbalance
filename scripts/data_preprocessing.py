import os
import glob
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Cấu hình đường dẫn 
TARGET_SIZE = 512

# Đường dẫn NIH
NIH_RAW_DIR = 'data/raw/nih-chest-xray'
NIH_OUT_DIR = 'data/processed/nih_512x512/images'

# Đường dẫn VinBigData
VIN_RAW_DIR = 'data/raw/vinbigdata-chest-xray/train'
VIN_OUT_DIR = 'data/processed/vin_512x512/images'

# Tạo sẵn các thư mục đầu ra
os.makedirs(NIH_OUT_DIR, exist_ok=True)
os.makedirs(VIN_OUT_DIR, exist_ok=True)

# Hàm xử lý ảnh VinBigData (DICOM-to-PNG), resize and save as .PNG
def process_vin(dicom_filename):
    input_path = os.path.join(VIN_RAW_DIR, dicom_filename)
    base_name = os.path.splitext(dicom_filename)[0] 
    output_path = os.path.join(VIN_OUT_DIR, f"{base_name}.png")

    if os.path.exists(output_path):
        return True

    try:
        dicom = pydicom.dcmread(input_path)
        img = apply_voi_lut(dicom.pixel_array, dicom)
        img = img.astype(np.float32)

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img

        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)

        img = np.uint8(img)
        img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, img_resized)
        return True
    except Exception as e:
        print(f"\nLỗi file Vin {dicom_filename}: {e}")
        return False

# Hàm xử lý NIH 
def process_nih(img_path):
    # Lấy tên file gốc (VD: 00000001_000.png)
    filename = os.path.basename(img_path)
    output_path = os.path.join(NIH_OUT_DIR, filename)

    if os.path.exists(output_path):
        return True

    try:
        # Đọc ảnh gốc ở chế độ Grayscale (Ảnh X-quang chỉ cần 1 kênh màu để lưu trữ cho nhẹ)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return False
            
        # Thu nhỏ ảnh (Dùng INTER_AREA cho việc thu nhỏ ảnh cho chất lượng tốt nhất)
        img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        
        # Lưu ảnh
        cv2.imwrite(output_path, img_resized)
        return True
    except Exception as e:
        print(f"\nLỗi file NIH {filename}: {e}")
        return False

# Luồng chạy chính 
def main():
    # Quét tìm toàn bộ file PNG của NIH (Đệ quy)
    print("Đang quét tìm file NIH...")
    nih_files = glob.glob(os.path.join(NIH_RAW_DIR, '**', '*.png'), recursive=True)
    print(f"Tìm thấy {len(nih_files)} file ảnh NIH.")

    # Quét tìm toàn bộ file DICOM của VinBigData
    print("Đang quét tìm file VinBigData...")
    vin_files = [f for f in os.listdir(VIN_RAW_DIR) if os.path.isfile(os.path.join(VIN_RAW_DIR, f))]
    print(f"Tìm thấy {len(vin_files)} file ảnh VinBigData.")

    num_cores = os.cpu_count()
    print(f"Bắt đầu xử lý với {num_cores} luồng CPU...")

    #Chạy xử lý NIH
    if len(nih_files) > 0:
        print(f"Đang xử lý {len(nih_files)} ảnh NIH Chest X-ray...")
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            list(tqdm(executor.map(process_nih, nih_files), total=len(nih_files), desc="NIH Progress"))

    # Chạy xử lý VinBigData
    if len(vin_files) > 0:
        print(f" Đang xử lý {len(vin_files)} ảnh VinBigData DICOM...")
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            list(tqdm(executor.map(process_vin, vin_files), total=len(vin_files), desc="VIN Progress"))

    print("Hoàn thành tiền xử lý")

if __name__ == "__main__":
    main()