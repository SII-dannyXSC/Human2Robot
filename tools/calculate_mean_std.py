import argparse
import json
import os
import numpy as np
import torch
import h5py
from omegaconf import OmegaConf
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--stats_path", type=str, default="./data/dataset_stats.pkl")
args = parser.parse_args()
config = OmegaConf.load(args.config)


def get_norm_stats(data_path_list):
    all_end_state_data = []
    all_end_position_data = []
    all_gripper_state_data = []
    all_action_data = []
    for dataset_path in data_path_list:
        have_action = False
        with h5py.File(dataset_path, "r") as root:
            end_position = root["/end_position"][()]
            gripper_state = root["/gripper_state"][()]
            end_state = np.concatenate([end_position, gripper_state[:, None]], axis=1)
            if "/action" in root:
                action = root["/action"][()]
                if action.shape[1] == 7:
                    have_action = True

        all_end_state_data.append(torch.from_numpy(end_state))
        if have_action:
            all_action_data.append(torch.from_numpy(action))
    if len(all_action_data) > 0:
        all_action_data = torch.vstack(all_action_data)

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
    else:
        action_std = torch.tensor(0)
        action_mean = torch.tensor(0)

    all_end_state_data = torch.vstack(all_end_state_data)
    # for end_state in all_end_state_data:
    #     if end_state[3] > 0:
    #         print(end_state)
    # normalize end_state data
    end_state_mean = all_end_state_data.mean(dim=0, keepdim=True)
    end_state_std = all_end_state_data.std(dim=0, keepdim=True)
    end_state_std = torch.clip(end_state_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "end_state_mean": end_state_mean.numpy().squeeze(),
        "end_state_std": end_state_std.numpy().squeeze(),
        "example_end_state": end_state,
    }

    return stats


if args.data_path:
    data_path_list = []
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.data_path):
        for name in files:
            if name.endswith(".hdf5"):
                video_mp4_paths.add(os.path.join(root, name))
    data_path_list = list(video_mp4_paths)
else:
    data_meta_paths = config.data.meta_paths
    vid_meta = []
    for data_meta_path in data_meta_paths:
        vid_meta.extend(json.load(open(data_meta_path, "r")))

    data_path_list = [item["data_path"] for item in vid_meta]
stats = get_norm_stats(data_path_list)
print(stats)

with open(args.stats_path, "wb") as f:
    pickle.dump(stats, f)
