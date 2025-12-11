import h5py
from omegaconf import OmegaConf
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

config = OmegaConf.load(
    "/home/ubuntu/project/Moore-AnimateAnyone/configs/train/stage2_3.yaml"
)
data_meta_paths = config.data.meta_paths
vid_meta = []
for data_meta_path in data_meta_paths:
    vid_meta.extend(json.load(open(data_meta_path, "r")))

data_path_list = [item["data_path"] for item in vid_meta]
cnt = 0
frame_number_list = []
for data_path in tqdm(data_path_list):
    try:
        with h5py.File(data_path, "r") as f:
            frame_number = f["timestamp"][()].shape[0]
            frame_number_list.append(frame_number)
            # human_video = np.uint8(f["/cam_data/human_camera"])
    except OSError:
        print("error: ", data_path)
        with open("./log.txt", "a") as f:
            f.write(data_path + "\n")
        break
print(frame_number_list)
plt.hist(frame_number_list, bins=10, edgecolor="black", color="skyblue")

# 添加标题和标签
plt.title("Frame Number Distribution")
plt.xlabel("Number of Frames")
plt.ylabel("Frequency")

# 显示图形
plt.show()
input()
