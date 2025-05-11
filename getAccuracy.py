import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras

# Load model
model = keras.models.load_model("helmet_detection_model.h5")

# Kích thước ảnh đầu vào
img_height = 224
img_width = 224

# Danh sách class theo thư mục
class_names = ['has_helmet', 'no_helmet']
class_indices = {name: idx for idx, name in enumerate(class_names)}

# Đường dẫn thư mục test
test_dir = r"C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\TestingData"

# Đếm đúng/sai và lưu thông tin ảnh sai
total = 0
correct = 0
wrong_predictions = []

for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue  # bỏ qua file không phải ảnh

        img_path = os.path.join(class_dir, fname)
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán
        prediction = model.predict(img_array, verbose=0)
        score = prediction[0][0]
        predicted_class_idx = int(score > 0.5)
        true_class_idx = class_indices[class_name]

        if predicted_class_idx == true_class_idx:
            correct += 1
        else:
            # Lưu thông tin ảnh sai
            wrong_predictions.append({
                'filename': fname,
                'true_class': class_name,
                'predicted_class': class_names[predicted_class_idx],
                'score': score
            })
        total += 1

# In kết quả tổng
accuracy = correct / total if total > 0 else 0
print(f"\n--- Accuracy Summary ---")
print(f"Total images         : {total}")
print(f"Correct predictions  : {correct}")
print(f"Wrong predictions    : {len(wrong_predictions)}")
print(f"Accuracy             : {accuracy:.4f}")

# In chi tiết các ảnh sai
if wrong_predictions:
    print(f"\n--- Misclassified Images ---")
    for i, info in enumerate(wrong_predictions, 1):
        print(f"{i}. {info['filename']}: True = {info['true_class']}, Predicted = {info['predicted_class']}, Score = {info['score']:.4f}")
else:
    print("\n🎉 Tất cả ảnh đều được phân loại đúng!")
