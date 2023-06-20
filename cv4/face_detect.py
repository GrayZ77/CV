import cv2
import face_recognition

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

face_match_threshold = 0.8  # 相似度阈值

while True:
    # 读取视频帧
    ret, frame = video.read()
    if not ret:
        break

    # 检测人脸
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 删除面积小于阈值的人脸
    faces = [face for face in faces if face[2] * face[3] > min_box_area]

    if frame_count == 1:
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
                    face_encoding = face_recognition.face_encodings(frame, [target])[0]
                    trackers[tracker] = id_counter  # 将跟踪器与ID进行映射
                    face_id[id_counter] = face_encoding
                    id_counter += 1

            face_detected = True

    # 如果没有进行人脸检测或当前帧为第660帧，则进行人脸检测
    if len(faces) > len(trackers):
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
                    face_encoding_current = face_recognition.face_encodings(frame, [target])[0]
                    id_current = 0
                    min_distance = 1
                    for id_former, face_encoding_former in face_id.items():
                        current_distance = face_recognition.face_distance([face_encoding_current], face_encoding_former)
                        if current_distance <= 0.6 and current_distance < min_distance:
                            # 匹配成功
                            min_distance = current_distance
                            id_current = id_former
                        else:
                            # 匹配不成功
                            trackers[tracker] = id_counter  # 将跟踪器与ID进行映射
                            face_id[id_counter] = face_encoding_current
                            id_counter += 1
                    trackers[tracker] = id_current  # 将跟踪器与ID进行映射
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
