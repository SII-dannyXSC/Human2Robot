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

root_path = os.path.join(args.root_path, args.meta_info_name)

# collect all video_folder paths
video_mp4_paths = set()
for root, dirs, files in os.walk(root_path):
    for name in files:
        if name.endswith(".hdf5"):
            video_mp4_paths.add(os.path.join(root, name))
video_mp4_paths = list(video_mp4_paths)

length = len(video_mp4_paths)
print(length)

meta_infos = []
for idx in range(length):
    video_mp4_path = video_mp4_paths[idx]
    meta_infos.append({"data_path": video_mp4_path})

json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
