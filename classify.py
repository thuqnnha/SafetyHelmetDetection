import os
import shutil
import xml.etree.ElementTree as ET

# Đường dẫn thư mục
img_dir = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\images'
xml_dir = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\annotations'
has_helmet_dir = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\has_helmet'
no_helmet_dir = r'C:\Users\FUJITSU\Desktop\SafetyHelmetDetection\no_helmet'

# Tạo thư mục nếu chưa có
os.makedirs(has_helmet_dir, exist_ok=True)
os.makedirs(no_helmet_dir, exist_ok=True)

# Duyệt từng file annotation
for filename in os.listdir(xml_dir):
    if not filename.endswith('.xml'):
        continue

    xml_path = os.path.join(xml_dir, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Kiểm tra xem có object nào là helmet hoặc head không
    has_helmet = False
    has_head = False

    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if name == 'helmet':
            has_helmet = True
        elif name == 'head':
            has_head = True

    # Tên ảnh tương ứng
    img_filename = root.find('filename').text
    img_path = os.path.join(img_dir, img_filename)

    if not os.path.exists(img_path):
        continue

    # Điều kiện chuyển ảnh
    if not has_helmet:
        shutil.copy(img_path, no_helmet_dir)
    elif not has_head:
        shutil.copy(img_path, has_helmet_dir)
