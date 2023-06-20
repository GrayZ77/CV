import cv2
from scipy.spatial.distance import cosine


def extract_facial_features(image_path):
    # 加载图像并创建人脸检测器和关键点检测器
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    facial_landmark_detector = cv2.face.createFacemarkLBF()

    # Load model
    facial_landmark_detector.loadModel('lbfmodel.yaml')

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 提取每个检测到的人脸的特征点
    facial_features = []
    ok, landmarks = facial_landmark_detector.fit(gray, faces)
    for landmark in landmarks:
        facial_features.append(landmark[0].flatten())  # 将特征点转换为一维数组

    return facial_features


def calculate_similarity(face1_features, face2_features):
    # 计算余弦相似度
    similarity_score = 1.0 - cosine(face1_features, face2_features)
    return similarity_score


# 图像路径
image1_path = 'frames_folder/frame_492.jpg'
image2_path = 'frames_folder/frame_661.jpg'

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载图像
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 灰度化图像
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 提取人脸特征
features1 = extract_facial_features(image1_path)
features2 = extract_facial_features(image2_path)

for i in range(len(features1)):
    match_num = 0
    max_similarity = 0
    for j in range(len(features2)):
        similarity = calculate_similarity(features1[i], features2[j])
        if similarity > 0.9 and similarity > max_similarity:
            max_similarity = similarity
            match_num = j
    print("similarity:", max_similarity)
    print("face", i + 1, "in image 1 matches face", match_num + 1, "in image 2")
    (x1, y1, w1, h1) = faces1[i]
    (x2, y2, w2, h2) = faces2[match_num]
    face1_image = image1[y1:y1 + h1, x1:x1 + w1]
    face2_image = image2[y2:y2 + h2, x2:x2 + w2]
    cv2.imwrite(f"face_matching/face_{i+1}_in_image_1.jpg", face1_image)
    cv2.imwrite(f"face_matching/face_{match_num+1}_in_image_2.jpg", face2_image)
