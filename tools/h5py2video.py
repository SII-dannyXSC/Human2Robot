import av
import numpy as np
import h5py
import argparse

"""
read hdf5 输出mp4
"""
parser = argparse.ArgumentParser(description="这是一个示例程序")

# 添加命令行参数
parser.add_argument("--input", type=str, help="输入文件路径", required=True)
parser.add_argument("--output", type=str, help="输出文件路径", required=True)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

hdf_file = args.input
output_file = args.output
fps = args.fps

width = 852
height = 240

# 创建输出容器
container = av.open(output_file, mode="w")

# 设置视频流（例如，设置分辨率和帧率）
stream = container.add_stream("libx264", rate=fps)
stream.width = width  # 设置宽度
stream.height = height  # 设置高度

# 打开 HDF5 文件
with h5py.File(hdf_file, "r") as f:
    video_data = f["/cam_data/human_camera"]
    robot_data = f["/cam_data/robot_camera"]

    for i in range(0, video_data.shape[0], 4):
        frame = video_data[i]
        frame = np.uint8(frame)
        frame2 = robot_data[i]
        frame2 = np.uint8(frame2)
        frame = np.hstack([frame, frame2])

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        # 将帧写入输出容器
        packet = stream.encode(av_frame)
        container.mux(packet)
# 完成编码并写入剩余帧
container.mux(stream.encode())

# 关闭容器，完成文件写入
container.close()

print(f"Video saved to {output_file}")
