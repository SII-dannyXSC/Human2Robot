import cv2


def save_frame(video_path, output_image_path, frame_number):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 设置要读取的帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 读取指定帧
    ret, frame = cap.read()

    if ret:
        # 如果成功读取到帧，则保存图像
        cv2.imwrite(output_image_path, frame)
        print(f"第 {frame_number} 帧已保存为 {output_image_path}")
    else:
        print(f"无法读取第 {frame_number} 帧")

    # 释放视频文件
    cap.release()


frame_number = 200  # 假设要读取第100帧

# 定义输入视频文件路径
# video_path = "./configs/inference/ref_images/episode_0.mp4"
video_path = "./configs/inference/pose_videos/episode_0.mp4"
# 定义输出图片文件路径
# output_image_path = f"./configs/inference/ref_images/episode_0_{frame_number}.png"
output_image_path = f"./configs/inference/pose_images/episode_0_{frame_number}.png"


save_frame(video_path, output_image_path, frame_number)
