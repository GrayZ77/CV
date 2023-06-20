import cv2


def save_frames_as_images(video_path, output_folder):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 确保视频文件被成功打开
    if not video.isOpened():
        print("无法打开视频文件。")
        return

    # 创建输出文件夹
    import os
    os.makedirs(output_folder, exist_ok=True)

    # 逐帧保存图像
    frame_count = 1
    while True:
        # 读取视频的下一帧
        success, frame = video.read()

        if not success:
            break

        # 构造输出文件路径
        output_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")

        # 保存图像
        cv2.imwrite(output_path, frame)

        frame_count += 1

    # 释放视频对象
    video.release()

    print(f"已保存 {frame_count} 帧图像到 {output_folder}。")


# 调用函数保存视频的每一帧为图片
video_path = "test.mp4"  # 替换为你的视频文件路径
output_folder = "frames_folder"  # 替换为你想保存图像的输出文件夹路径
save_frames_as_images(video_path, output_folder)
