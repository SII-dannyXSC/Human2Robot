import argparse
import json
import os

# -----
# [{'vid': , 'kps': , 'other':},
#  {'vid': , 'kps': , 'other':}]
# -----
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
# -----
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

# pose_dir = args.root_path + "_dwpose"
human_root_path = os.path.join(args.root_path,"human")
robot_root_path = os.path.join(args.root_path,"robot")

# collect all video_folder paths
human_mp4_paths = set()
for root, dirs, files in os.walk(human_root_path):
    for name in files:
        if name.endswith(".mp4"):
            human_mp4_paths.add(os.path.join(root, name))
human_mp4_paths = list(human_mp4_paths)

meta_infos = []
for human_path in human_mp4_paths:
    relative_video_name = os.path.relpath(human_path, human_root_path)
    robot_path = os.path.join(robot_root_path, relative_video_name)
    meta_infos.append({"human_path": human_path, "robot_path": robot_path})

json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
