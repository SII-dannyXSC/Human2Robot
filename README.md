<div align="center" style="font-family: charter;">
<h1>Human2Robot: Learning Robot Actions from Paired Human-Robot Videos</h1>
<a href="https://arxiv.org/abs/2502.16587" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-H2R-red?logo=arxiv" height="20" />
</a>
<a href="https://huggingface.co/datasets/dannyXSC/HumanAndRobot" target="_blank">
    <img alt="HF Dataset: SPEC" src="https://img.shields.io/badge/📒_Dataset-H2R-ffc107?color=A9B5DF&logoColor=white" height="20" />
</a>
<div>
    Sicheng Xie<sup>*</sup>,</span>
    Haidong Cao<sup>*</sup>,</span>
    Zejia Weng<sup></sup>,</span>
    Zhen Xing<sup></sup>,</span>
    Haoran Chen<sup></sup>,</span>
    Shiwei Shen<sup></sup>,</span>
    Jiaqi Leng<sup></sup>,<br/></span>
    <a href="https://zxwu.azurewebsites.net/" target="_blank">Zuxuan Wu</a><sup>&dagger;</sup>,</span>
    Yu-Gang Jiang<sup></sup></span>
</div>

<div>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>&dagger;</sup> Corresponding author&emsp;
</div>

</div>

<p align="center" style="margin: 4px 0 12px 0; color: #444;">
    <em>Accepted as Oral at AAAI 2026.</em>
</p>

<div align="center">
    <img src="assets/method.jpg" alt="Method overview diagram" width="720">
</div>


## Build Environment

```
conda create -n h2r python=3.10
conda activate h2r

pip install pip==24.0.0
pip install -r requirements.txt

# install pytorch according to your CUDA version
# e.g., pip install torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

python tools/download_weights.py
```

## Prepare the Dataset

Download the HumanAndRobot dataset:

[https://huggingface.co/dannyXSC/HumanAndRobot](https://huggingface.co/dannyXSC/HumanAndRobot)

Place it under:

```
./data/HumanAndRobot/
```

## Training and Inference

### Stage 1

Download the following pretrained models and place them under:

```
./pretrained_weights/
```

1. OpenPose ControlNet
   [https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/tree/main](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/tree/main)

2. SD Image Variation (UNet initialization)
   [https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main)

Run Stage 1 training:

```
accelerate launch train_stage_1.py --config configs/train/stage1.yaml
```

### Stage 2

Download motion module weights:

mm_sd_v15_v2.ckpt
[https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)

Place it under:

```
./pretrained_weights/
```

Specify Stage 1 weights in `configs/train/stage2.yaml`:

```
stage1_ckpt_dir: './exp_output/stage1'
stage1_ckpt_step: 30000
```

Run Stage 2 training:

```
accelerate launch train_stage_2.py --config configs/train/stage2.yaml
```

## TODO List

### Completed

* Dataset release (HumanAndRobot)
* Training pipeline for Stage 1 and Stage 2
* Video generation model implementation
* Pretrained weight download script
* Training scripts and configurations

### In Progress / Not Completed

* Data preprocessing scripts
* Imitation learning (IL) training code
* Inference code and generation examples
* Demo / WebUI
* Model weight release (optional)

## Citation

If you use this project or dataset, please cite:

```
@article{xie2025human2robot,
  title={Human2robot: Learning robot actions from paired human-robot videos},
  author={Xie, Sicheng and Cao, Haidong and Weng, Zejia and Xing, Zhen and Chen, Haoran and Shen, Shiwei and Leng, Jiaqi and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2502.16587},
  year={2025}
}
```

## Acknowledgements

This project is built upon ControlNet, Stable Diffusion, AnimateDiff, Diffusers, and the HumanAndRobot dataset.
