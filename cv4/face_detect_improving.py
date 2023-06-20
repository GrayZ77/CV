import cv2
from scipy.spatial.distance import cosine


def extract_facial_features(faces):
    facial_landmark_detector = cv2.face.createFacemarkLBF()
    # Loading model
    facial_landmark_detector.loadModel('lbfmodel.yaml')

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


# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 创建目标跟踪器字典
trackers = {}
id_counter = 1

# 加载视频文件
video = cv2.VideoCapture('test.mp4')

# 获取视频的帧率和尺寸
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
result_video = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 读取初始帧
ret, frame = video.read()
if not ret:
    exit()

face_id = {}

# 初始化人脸检测和目标跟踪
face_detected = False

# 当前帧数
frame_count = 1

# 设定最小目标框面积阈值
min_box_area = 10000

while True:
    # 读取视频帧
    ret, frame = video.read()
    if not ret:
        break

    # 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if frame_count == 1:
        features = extract_facial_features(faces)
        if len(faces) > 0:
            # 清空之前的目标跟踪器字典
            trackers = {}

            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                target = (x, y, w, h)
                area = w * h
                # 如果目标框面积大于设定阈值，则创建新的目标跟踪器
                if area > min_box_area:
                    tracker = cv2.TrackerKCF_create()  # 创建新的目标跟踪器
                    tracker.init(frame, target)  # 初始化新的目标跟踪器
                    trackers[tracker] = id_counter  # 将跟踪器与ID进行映射
                    face_id[id_counter] = features[i]
                    id_counter += 1

            face_detected = True

    # 如果没有进行人脸检测或当前帧为第660帧，则进行人脸检测
    if len(faces) > len(trackers):
        features = extract_facial_features(faces)
        if len(faces) > 0:
            # 清空之前的目标跟踪器字典
            trackers = {}

            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                target = (x, y, w, h)
                area = w * h
                # 如果目标框面积大于设定阈值，则创建新的目标跟踪器
                if area > min_box_area:
                    tracker = cv2.TrackerKCF_create()  # 创建新的目标跟踪器
                    tracker.init(frame, target)  # 初始化新的目标跟踪器
                    feature_current = features[i]
                    id_current = 0
                    similarity_max = 0
                    for id_former, feature_former in face_id.items():
                        similarity = calculate_similarity(feature_current, feature_former)
                        if similarity > similarity_max and similarity > 0.8:
                            similarity_max = similarity
                            id_current = id_former
                    trackers[tracker] = id_current  # 将跟踪器与ID进行映射
                    id_counter += 1
            face_detected = True

    # 更新目标位置
    trackers_to_delete = []  # 存储需要删除的跟踪器
    for tracker in trackers.keys():
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            tracker_id = trackers[tracker]  # 获取跟踪器对应的ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(tracker_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # 跟踪失败，标记为需要删除
            trackers_to_delete.append(tracker)

    # 删除需要删除的跟踪器
    for tracker in trackers_to_delete:
        del trackers[tracker]

    # 如果没有进行目标跟踪，则重新进行人脸检测
    if len(trackers) == 0:
        face_detected = False

    # 将当前帧写入结果视频
    result_video.write(frame)

    cv2.imshow("Tracking", frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# 释放视频对象和窗口
video.release()
result_video.release()
cv2.destroyAllWindows()
