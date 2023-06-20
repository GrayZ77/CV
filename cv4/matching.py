import face_recognition
from PIL import Image


def face_matching(frame1_path, frame2_path):
    # 加载图片
    frame1 = face_recognition.load_image_file(frame1_path)
    frame2 = face_recognition.load_image_file(frame2_path)

    # 提取人脸编码
    encodings1 = face_recognition.face_encodings(frame1)
    encodings2 = face_recognition.face_encodings(frame2)

    # 检查是否存在足够的人脸进行匹配
    if len(encodings1) < 2 or len(encodings2) < 2:
        print("图片中人脸数量不足，无法进行匹配")
        return

    # 比较人脸编码
    match_1 = face_recognition.compare_faces([encodings1[0]], encodings2[0])[0]
    match_2 = face_recognition.compare_faces([encodings1[1]], encodings2[1])[0]

    # 输出匹配结果
    print("人脸1匹配结果:", "成功" if match_1 else "失败")
    print("人脸2匹配结果:", "成功" if match_2 else "失败")

    # 保存人脸图像
    if match_1:
        face1_image = frame1[
                      face_recognition.face_locations(frame1)[0][0]:face_recognition.face_locations(frame1)[0][2],
                      face_recognition.face_locations(frame1)[0][3]:face_recognition.face_locations(frame1)[0][1]]
        pil_image1 = Image.fromarray(face1_image)
        pil_image1.save("image1_face1.jpg")
        print("人脸图像已保存为：image1_face1.jpg")

    if match_2:
        face2_image = frame1[
                      face_recognition.face_locations(frame1)[1][0]:face_recognition.face_locations(frame1)[1][2],
                      face_recognition.face_locations(frame1)[1][3]:face_recognition.face_locations(frame1)[1][1]]
        pil_image2 = Image.fromarray(face2_image)
        pil_image2.save("image1_face2.jpg")
        print("人脸图像已保存为：image1_face2.jpg")

    if match_1 and len(encodings2) > 1:
        face3_image = frame2[
                      face_recognition.face_locations(frame2)[0][0]:face_recognition.face_locations(frame2)[0][2],
                      face_recognition.face_locations(frame2)[0][3]:face_recognition.face_locations(frame2)[0][1]]
        pil_image3 = Image.fromarray(face3_image)
        pil_image3.save("image2_face1.jpg")
        print("人脸图像已保存为：image2_face1.jpg")

    if match_2 and len(encodings2) > 1:
        face4_image = frame2[
                      face_recognition.face_locations(frame2)[1][0]:face_recognition.face_locations(frame2)[1][2],
                      face_recognition.face_locations(frame2)[1][3]:face_recognition.face_locations(frame2)[1][1]]
        pil_image4 = Image.fromarray(face4_image)
        pil_image4.save("image2_face2.jpg")
        print("人脸图像已保存为：image2_face2.jpg")


# 测试代码
frame1_path = "frame_493.jpg"  # 替换为frame1.jpg的实际路径
frame2_path = "frame_662.jpg"  # 替换为frame2.jpg的实际路径

face_matching(frame1_path, frame2_path)
