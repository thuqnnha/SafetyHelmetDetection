from ultralytics import YOLO
import os
import cv2

# Đường dẫn
input_folder = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\only_no_helmet'  # folder chứa ảnh gốc
output_folder = 'person'  # folder lưu ảnh person crop

# Tạo folder output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Load model YOLOv8 (model mặc định detect 80 lớp COCO, trong đó person là id=0)
model = YOLO('yolov8x.pt')  
# Lặp qua từng ảnh trong thư mục
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Detect
    results = model.predict(img, classes=[0], conf=0.3)  # chỉ detect class 0 (person), conf threshold = 0.3

    # results[0].boxes.xyxy chứa toạ độ các box [x1, y1, x2, y2]
    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)

        # Crop ảnh
        crop = img[y1:y2, x1:x2]

        # Lưu ảnh crop
        output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_person_{idx}.jpg")
        cv2.imwrite(output_path, crop)

print("Xong! Đã lưu các ảnh person đã crop.")
