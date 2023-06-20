import cv2
import numpy as np


# def extract_color_histogram(image_path, num_bins=8):
#     # 读取图像
#     image = cv2.imread(image_path)
#     # 将图像转换为RGB颜色空间
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # 计算全局RGB直方图
#     hist = cv2.calcHist([image], [0, 1, 2], None, [num_bins, num_bins, num_bins], [0, 256, 0, 256, 0, 256])
#     # 归一化直方图
#     hist = cv2.normalize(hist, hist).flatten()
#     return hist
#
#
# # 用于计算两个直方图之间的巴氏距离（可选的相似度度量）
# def bhattacharyya_distance(hist1, hist2):
#     return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


# 保存结果的数据结构
class Result:
    def __init__(self, image_path, similarity):
        self.image_path = image_path
        self.similarity = similarity


# 需要检索的图像列表
# image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg",
#                "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg", "21.jpg",
#                "22.jpg", "23.jpg", "24.jpg", "25.jpg", "26.jpg", "27.jpg", "28.jpg", "29.jpg", "30.jpg", "31.jpg",
#                "32.jpg", "33.jpg", "34.jpg", "35.jpg", "36.jpg", "37.jpg", "38.jpg", "39.jpg", "40.jpg", "41.jpg",
#                "42.jpg", "43.jpg", "44.jpg", "45.jpg", "46.jpg", "47.jpg", "48.jpg", "49.jpg", "50.jpg"]


# def extract_color_moments(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#     # 将图像转换为HSV颜色空间
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # 提取H、S、V通道
#     h, s, v = cv2.split(hsv_image)
#     # 计算颜色矩特征
#     h_mean = np.mean(h)
#     s_mean = np.mean(s)
#     v_mean = np.mean(v)
#     # 提取颜色矩特征向量
#     features = np.array([h_mean, s_mean, v_mean])
#     return features
#
#
# # 用于计算两个特征向量之间的欧几里得距离（可选的相似度度量）
# def euclidean_distance(features1, features2):
#     distance = np.linalg.norm(features1 - features2)
#     similarity = 1 / (1 + distance)
#     return similarity


# for image_single in image_paths:
#     # 提取当前图像的全局RGB直方图
#     hist = extract_color_histogram("./game_image/" + image_single)
#     # 计算查询图像与当前图像之间的相似度（巴氏距离）
#     similarity = bhattacharyya_distance(query_hist, hist)
#     # 创建结果对象并添加到结果列表中
#     result = Result(image_single, similarity)
#     results.append(result)
# query_hist = extract_color_histogram(file_read)

# for i in range(5):
#     results = []
#     file_read = f"./game_image/source{i + 1}.jpg"
#     # 提取要查询的图像的全局RGB直方图
#
#     query_features = extract_color_moments(file_read)
#
#     # 遍历图像列表并计算相似度
#     for image_path in image_paths:
#         # 提取当前图像的颜色矩特征
#         features = extract_color_moments("./game_image/" + image_path)
#         # 计算查询图像与当前图像之间的相似度（欧几里得距离）
#         similarity = euclidean_distance(query_features, features)
#         # 创建结果对象并添加到结果列表中
#         result = Result(image_path, similarity)
#         results.append(result)
#     # 根据相似度排序结果
#     results.sort(key=lambda x: x.similarity, reverse=True)
#     # 保存前3个结果的图像路径和相似度
#     top_results = results[:3]
#     print('Source Image:', file_read)
#     # 打印前3个结果
#     for result in top_results:
#         print('Image:', result.image_path)
#         print('Similarity:', result.similarity)
#         print('---')
#     print()

target_images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg",
                 "12.jpg", "13.jpg", "14.jpg", "15.jpg", "16.jpg", "17.jpg", "18.jpg", "19.jpg", "20.jpg", "21.jpg",
                 "22.jpg", "23.jpg", "24.jpg", "25.jpg", "26.jpg", "27.jpg", "28.jpg", "29.jpg", "30.jpg", "31.jpg",
                 "32.jpg", "33.jpg", "34.jpg", "35.jpg", "36.jpg", "37.jpg", "38.jpg", "39.jpg", "40.jpg", "41.jpg",
                 "42.jpg", "43.jpg", "44.jpg", "45.jpg", "46.jpg", "47.jpg", "48.jpg", "49.jpg", "50.jpg"]

for i in range(5):
    file_read = f"./game_image/source{i + 1}.jpg"
    sift = cv2.SIFT_create()
    target_image = cv2.imread(file_read, 0)
    target_kp, target_des = sift.detectAndCompute(target_image, None)
    similarities = []
    for image_path in target_images:
        candidate_image = cv2.imread("./game_image/" + image_path, 0)
        candidate_kp, candidate_des = sift.detectAndCompute(candidate_image, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(target_des, candidate_des, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        similarity = len(good_matches)
        similarities.append((similarity, image_path))
    similarities.sort(reverse=True)
    similar_images = [image_path for similarity, image_path in similarities[:3]]
    print("最相似的三个图像：")
    for image_path in similar_images:
        print(image_path)
