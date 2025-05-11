from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load model
model = keras.models.load_model("helmet_detection_model.h5")

# Load ảnh
img_path = r"C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\TestingData\no_helmet\no_helmet10.jpg"
img_height = 224
img_width = 224

img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

# KHÔNG cần chia 255 nếu model đã có layers.Rescaling(1./255)
# img_array /= 255.0  <-- hãy COMMENT dòng này lại nếu model có Rescaling

# Dự đoán
prediction = model.predict(img_array)
score = prediction[0][0]

# Danh sách class
class_names = ['has_helmet', 'no_helmet']
predicted_class = class_names[int(score > 0.5)]

print(f"Prediction raw value: {score:.4f}")
print(f"Predicted class: {predicted_class}")

# Hiển thị ảnh
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {predicted_class} ({score:.2f})")
plt.show()
