import face_recognition
import cv2
import os
import pickle
from imutils import paths

# Đường dẫn tới thư mục chứa hình ảnh
imagePaths = list(paths.list_images("dataset"))

# Tạo các danh sách để lưu mã hóa và tên
knownEncodings = []
knownNames = []

# Duyệt qua từng hình ảnh
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i+1}/{len(imagePaths)}")
    
    # Trích xuất tên của người từ đường dẫn (lấy tên thư mục)
    name = imagePath.split(os.path.sep)[-2]

    # Đọc ảnh và chuyển sang định dạng RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Xác định các vùng chứa khuôn mặt
    boxes = face_recognition.face_locations(rgb, model="hog")

    # Mã hóa khuôn mặt
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Lưu trữ các mã hóa và tên
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# Lưu các mã hóa và tên vào file
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))
