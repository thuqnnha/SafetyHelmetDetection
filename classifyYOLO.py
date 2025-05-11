import os
import shutil

# Đường dẫn thư mục
image_folder = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\dataRoboflow\train\images'
label_folder = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\dataRoboflow\train\labels'

# Tạo thư mục lưu ảnh kết quả
only_has_helmet_folder = 'only_has_helmet'
only_no_helmet_folder = 'only_no_helmet'
os.makedirs(only_has_helmet_folder, exist_ok=True)
os.makedirs(only_no_helmet_folder, exist_ok=True)

# Lặp qua tất cả file nhãn
for label_file in os.listdir(label_folder):
    if not label_file.endswith('.txt'):
        continue

    label_path = os.path.join(label_folder, label_file)
    image_name = os.path.splitext(label_file)[0]
    
    # Tìm file ảnh tương ứng
    for ext in ['.jpg', '.png', '.jpeg']:
        image_path = os.path.join(image_folder, image_name + ext)
        if os.path.exists(image_path):
            break
    else:
        print(f"Không tìm thấy ảnh tương ứng cho {label_file}")
        continue

    # Đọc class_id từ file nhãn
    with open(label_path, 'r') as f:
        class_ids = [int(line.strip().split()[0]) for line in f if line.strip()]

    # Kiểm tra nếu tất cả là người đội mũ
    if all(cid == 0 for cid in class_ids):
        shutil.copy(image_path, only_has_helmet_folder)
    # Nếu tất cả là người không đội mũ
    elif all(cid == 1 for cid in class_ids):
        shutil.copy(image_path, only_no_helmet_folder)
