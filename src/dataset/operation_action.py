import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import h5py
import torchvision.transforms.functional as TF
import pickle


class OperationVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        stats_path,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # TODO: condition需要pixel方式吗
        self.cond_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

        # for normalization
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        # self.pre_process = (
        #     lambda s_end_state: (s_end_state - stats["end_state_mean"])
        #     / stats["end_state_std"]
        # )

    def pre_process(self, s_end_state):
        s_end_state[..., :3] = (
            s_end_state[..., :3] - self.stats["end_state_mean"][:3]
        ) / self.stats["end_state_std"][:3]
        # angle
        s_end_state[..., 3:6] = s_end_state[..., 3:6] / 180
        # gripper not change
        # s_end_state[..., 7]
        return s_end_state

    def same_crop(self, image_list):
        if len(image_list) == 0:
            return

        for k in range(len(image_list)):
            width, height = image_list[k].size

            pad_width = max(0, self.width - width)
            pad_height = max(0, self.height - height)

            # 如果需要填充，则进行填充
            if pad_width > 0 or pad_height > 0:
                padding_transform = transforms.Pad(
                    (
                        pad_width // 2,
                        pad_height // 2,
                        pad_width - pad_width // 2,
                        pad_height - pad_height // 2,
                    ),
                    fill=(0, 0, 0),
                )
                image_list[k] = padding_transform(image_list[k])

        i, j, h, w = transforms.RandomCrop.get_params(
            image_list[0], output_size=(self.height, self.width)
        )
        for k in range(len(image_list)):
            image_list[k] = TF.crop(image_list[k], i, j, h, w)

        return image_list

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        data_path = video_meta["data_path"]

        with h5py.File(data_path, "r") as f:
            human_video = np.uint8(f["/cam_data/human_camera"])
            robot_video = np.uint8(f["/cam_data/robot_camera"])

            end_position = f["/end_position"][()]
            gripper_state = f["/gripper_state"][()]
            end_state = np.concatenate([end_position, gripper_state[:, None]], axis=1)
            end_state = self.pre_process(end_state)
            # action = f["/action"][()]

            # BGR -> RGB
            if human_video.shape[3] == 3:
                human_video = human_video[:, :, :, ::-1]
            if robot_video.shape[3] == 3:
                robot_video = robot_video[:, :, :, ::-1]

        video_length = len(human_video)
        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # set ref img as start frame of robot video
        ref_img_idx = start_idx
        ref_img = Image.fromarray(robot_video[ref_img_idx])
        ref_pose_img = Image.fromarray(human_video[ref_img_idx])

        # read frames and kps
        human_image_list = []
        robot_image_list = []
        ref_pose_image_list = []
        end_state_list = []
        for index in batch_index:
            human_img = human_video[index]
            human_image_list.append(Image.fromarray(human_img))
            robot_img = robot_video[index]
            robot_image_list.append(Image.fromarray(robot_img))
            ref_pose_image_list.append(ref_pose_img)

            end_state_list.append(end_state[index])

        result = self.same_crop(
            human_image_list + robot_image_list + ref_pose_image_list
        )
        human_image_list = result[: self.n_sample_frames]
        robot_image_list = result[self.n_sample_frames : self.n_sample_frames * 2]
        ref_pose_image_list = result[self.n_sample_frames * 2 :]

        # transform
        state = torch.get_rng_state()
        pixel_values_robot = self.augmentation(
            robot_image_list, self.pixel_transform, state
        )
        pixel_values_human_ref = self.augmentation(
            ref_pose_image_list, self.cond_transform, state
        )
        pixel_values_human = self.augmentation(
            human_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        end_state_list = torch.from_numpy(np.array(end_state_list)).float()

        sample = dict(
            video_dir=data_path,
            pixel_values_vid=pixel_values_robot,
            pixel_values_pose_ref=pixel_values_human_ref,
            pixel_values_pose=pixel_values_human,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
            end_state_list=end_state_list,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)


if __name__ == "__main__":
    dataset = OperationVideoDataset(
        4,
        24,
        424,
        240,
        "/home/ubuntu/project/Moore-AnimateAnyone/data/dataset_stats.pkl",
        data_meta_paths=[
            "/home/ubuntu/project/Moore-AnimateAnyone/data/grab_cube_meta.json"
        ],
    )
    dataset[0]
