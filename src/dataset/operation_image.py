import json
import random
import h5py

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import numpy as np
import torchvision.transforms.functional as TF


class OperationDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
    ):
        super().__init__()

        # width height
        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     self.img_size,
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     self.img_size,
                #     scale=self.img_scale,
                #     ratio=self.img_ratio,
                #     interpolation=transforms.InterpolationMode.BILINEAR,
                # ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def same_crop(self, image_list):
        if len(image_list) == 0:
            return

        for k in range(len(image_list)):
            width, height = image_list[k].size

            pad_width = max(0, self.img_size[0] - width)
            pad_height = max(0, self.img_size[1] - height)

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
            image_list[0], output_size=(self.img_size[1], self.img_size[0])
        )
        for k in range(len(image_list)):
            image_list[k] = TF.crop(image_list[k], i, j, h, w)

        return image_list

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        data_path = video_meta["data_path"]

        with h5py.File(data_path, "r") as f:
            human_video = np.uint8(f["/cam_data/human_camera"])
            robot_video = np.uint8(f["/cam_data/robot_camera"])
            # BGR -> RGB
            if human_video.shape[3] == 3:
                human_video = human_video[:, :, :, ::-1]
            if robot_video.shape[3] == 3:
                robot_video = robot_video[:, :, :, ::-1]

        video_length = len(human_video)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - margin)
        tgt_img_idx = random.randint(
            ref_img_idx, min(video_length - 1, ref_img_idx + margin)
        )

        ref_img = robot_video[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img)
        tgt_img = robot_video[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img)

        src_img = human_video[tgt_img_idx]
        src_img_pil = Image.fromarray(src_img)

        [tgt_img_pil, src_img_pil, ref_img_pil] = self.same_crop(
            [tgt_img_pil, src_img_pil, ref_img_pil]
        )

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        src_img_img = self.augmentation(src_img_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=data_path,
            img=tgt_img,
            tgt_pose=src_img_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        return sample

    # def __getitem__(self, index):
    #     video_meta = self.vid_meta[index]
    #     human_video_path = video_meta["human_path"]
    #     robot_video_path = video_meta["robot_path"]

    #     human_reader = VideoReader(human_video_path)
    #     robot_reader = VideoReader(robot_video_path)

    #     assert len(human_reader) == len(
    #         robot_reader
    #     ), f"{len(human_reader) = } != {len(robot_reader) = } in {human_video_path}"

    #     video_length = len(human_reader)

    #     margin = min(self.sample_margin, video_length - 1)

    #     ref_img_idx = random.randint(0, video_length - 1 - margin)
    #     tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)

    #     ref_img = robot_reader[ref_img_idx]
    #     ref_img_pil = Image.fromarray(ref_img.asnumpy())
    #     tgt_img = robot_reader[tgt_img_idx]
    #     tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

    #     src_img = human_reader[tgt_img_idx]
    #     src_img_pil = Image.fromarray(src_img.asnumpy())

    #     [tgt_img_pil, src_img_pil, ref_img_pil] = self.same_crop(
    #         [tgt_img_pil, src_img_pil, ref_img_pil]
    #     )

    #     state = torch.get_rng_state()
    #     tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
    #     src_img_img = self.augmentation(src_img_pil, self.cond_transform, state)
    #     ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
    #     clip_image = self.clip_image_processor(
    #         images=ref_img_pil, return_tensors="pt"
    #     ).pixel_values[0]

    #     sample = dict(
    #         video_dir=human_video_path,
    #         img=tgt_img,
    #         tgt_pose=src_img_img,
    #         ref_img=ref_img_vae,
    #         clip_images=clip_image,
    #     )

    #     return sample

    def __len__(self):
        return len(self.vid_meta)


if __name__ == "__main__":
    dataset = OperationDataset(
        (256, 256),
        data_meta_paths=[
            "/home/ubuntu/project/Moore-AnimateAnyone/data/None_meta.json"
        ],
    )
    dataset[0]
    # def tensor_to_image(tensor_image, path):
    #     tensor_image = tensor_image.permute(1, 2, 0)  # 转换为 (H, W, C)

    #     # 将张量的数值范围从 [0, 1] 转换为 [0, 255]
    #     tensor_image = (
    #         (tensor_image * 255).clamp(0, 255).byte()
    #     )  # 转换为 uint8 类型

    #     # 将张量转换为 PIL 图像
    #     image = Image.fromarray(tensor_image.numpy())

    #     # 保存图像
    #     image.save(path)

    # tensor_to_image(tgt_img, "tgt_img.png")
    # tensor_to_image(src_img_img, "src_img_img.png")
    # tensor_to_image(ref_img_vae, "ref_img_vae.png")
    # tgt_img_pil.save("tgt_img_pil.png")
    # src_img_pil.save("src_img_pil.png")
    # ref_img_pil.save("ref_img_pil.png")
    # exit()
