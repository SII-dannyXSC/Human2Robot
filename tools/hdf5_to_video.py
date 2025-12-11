import cv2
import h5py
import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--save_path", type=str)

args = parser.parse_args()


def save_video(frames, save_path):
    print(f"saving :{save_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (426, 240))

    for i in range(frames.shape[0]):
        frame = np.uint8(frames[i])
        out.write(frame)

    out.release()

    print(f"saving :{save_path} complete!")


def hdf5_to_video(path, name, save_path):
    # 打开 HDF5 文件
    with h5py.File(path, "r") as f:
        human_data = f["/cam_data/human_camera"]
        robot_data = f["/cam_data/robot_camera"]

        save_video(human_data, os.path.join(save_path, "human", name))
        save_video(robot_data, os.path.join(save_path, "robot", name))


def find_unique_files(folder_hdf5, folder_mp4):
    # 获取文件夹a中的所有文件名
    files_in_a = set([name.rstrip(".hdf5") for name in os.listdir(folder_hdf5)])

    # 获取文件夹b中的所有文件名
    files_in_b = set([name.rstrip(".mp4") for name in os.listdir(folder_mp4)])

    # 找出文件夹a中存在但文件夹b中不存在的文件
    unique_files_in_a = files_in_a - files_in_b

    return unique_files_in_a


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

human_path = os.path.join(args.save_path, "human")
if not os.path.exists(human_path):
    os.makedirs(human_path)

robot_path = os.path.join(args.save_path, "robot")
if not os.path.exists(robot_path):
    os.makedirs(robot_path)


# collect all video_folder paths
video_mp4_paths = set()
for root, dirs, files in os.walk(args.root_path):
    for name in files:
        if name.endswith(".hdf5"):
            tgt_name = name.rstrip(".hdf5") + ".mp4"
            hdf5_to_video(os.path.join(root, name), tgt_name, args.save_path)
