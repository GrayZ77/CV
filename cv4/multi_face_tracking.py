import cv2


# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 创建目标跟踪器字典
trackers = {}

# 加载视频文件
video = cv2.VideoCapture('test.mp4')

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

    # 删除面积小于阈值的人脸
    faces = [face for face in faces if face[2] * face[3] > min_box_area]

    if frame_count == 1 or len(faces) > len(trackers):
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
                    trackers[tracker] = target

            face_detected = True

    # 更新目标位置
    trackers_to_delete = []  # 存储需要删除的跟踪器
    for tracker in trackers.keys():
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            target = (x, y, w, h)
            trackers[tracker] = target  # 更新目标位置
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # 跟踪失败，标记为需要删除
            trackers_to_delete.append(tracker)

    # 删除需要删除的跟踪器
    for tracker in trackers_to_delete:
        del trackers[tracker]

    # 如果没有进行目标跟踪，则重新进行人脸检测
    if len(trackers) == 0:
        face_detected = False

    cv2.imshow("Tracking", frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imwrite(f'face_tracking/frame_{frame_count}.jpg', frame)
    frame_count += 1

video.release()
cv2.destroyAllWindows()
