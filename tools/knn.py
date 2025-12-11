import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import h5py
import numpy as np
import os
from tqdm import tqdm
import io
from abc import ABC, abstractmethod
import pickle
from transformers import CLIPProcessor, CLIPModel


def feature_extractor_factory(backbone_name="resnet", normalize=False, device="cuda"):
    SUPPORT_BACKBONE = set(["resnet", "dinov2", "clip", "dinov2+clip"])
    assert backbone_name in SUPPORT_BACKBONE, "feature extractor is not supported."

    if backbone_name == "resnet":
        return ResnetFeatureExtractor(normalize=normalize, device=device)
    elif backbone_name == "dinov2":
        return Dinov2FeatureExtractor(normalize=normalize, device=device)
    elif backbone_name == "clip":
        return CLIPFeatureExtractor(normalize=normalize, device=device)
    elif backbone_name == "dinov2+clip":
        return Dinov2CLIPFeatureExtractor(normalize=normalize, device=device)


class FeatureExtractor(ABC):
    def close(self):
        if next(self.model.parameters()).is_cuda:
            del self.model
            torch.cuda.empty_cache()


class Dinov2CLIPFeatureExtractor(FeatureExtractor):
    def __init__(self, normalize=False, device="cuda"):
        self.clip = CLIPFeatureExtractor(normalize, device=device)
        self.dinov2 = Dinov2FeatureExtractor(normalize, device=device)

    def __call__(self, image_pil):
        f1 = self.clip(image_pil)
        f2 = self.dinov2(image_pil)

        return torch.cat([f1, f2], dim=-1)

    def close(self):
        self.clip.close()
        self.dinov2.close()


class CLIPFeatureExtractor(FeatureExtractor):
    def __init__(self, normalize=False, device="cuda"):
        super().__init__()
        model_path = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_path)
        self.model.eval()
        self.model.to(device)

        self.preprocess = CLIPProcessor.from_pretrained(model_path)
        self.normalize = normalize
        self.device = device

    def __call__(self, image_pil):
        inputs = self.preprocess(images=image_pil, return_tensors="pt")
        inputs = inputs.to(self.device)

        # 提取特征
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            if self.normalize:
                features = F.normalize(features, p=2, dim=-1)
        # 1 768
        return features.to("cpu")


class Dinov2FeatureExtractor(FeatureExtractor):

    def __init__(self, normalize=False, device="cuda"):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.model.eval()
        self.model.to(device)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.normalize = normalize
        self.device = device

    def __call__(self, image_pil):
        image = self.preprocess(image_pil).unsqueeze(0)  # 增加batch维度
        image = image.to(self.device)

        # 提取特征
        with torch.no_grad():
            features = self.model(image)
            if self.normalize:
                features = F.normalize(features, p=2, dim=-1)
        # 1 1024
        return features.to("cpu")


class ResnetFeatureExtractor(FeatureExtractor):

    def __init__(self, normalize=False, device="cuda"):
        # 加载预训练的ResNet50模型
        resnet = models.resnet50(pretrained=True)
        # 去掉最后一层
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval()  # 设置为评估模式
        self.model.to(device)

        # 定义预处理函数
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.normalize = normalize
        self.device = device

    def __call__(self, image_pil):
        image = self.preprocess(image_pil).unsqueeze(0)  # 增加batch维度
        image = image.to(self.device)

        # 提取特征
        with torch.no_grad():
            features = self.model(image)

        # 转换为一维特征向量
        # 1 2048
        features = features.view(features.size(0), -1)
        if self.normalize:
            features = F.normalize(features, p=2, dim=-1)
        return features.to("cpu")


class ImageDatabase:
    SUPPORT_INDEX_TYPE = set(["L2", "IP", "COS"])

    def __init__(self, backbone_name="resnet", index_type="L2", device="cuda"):
        assert index_type in self.SUPPORT_INDEX_TYPE, "index type is not supported."
        self.index = None
        self.backbone_name = backbone_name
        self.index_type = index_type
        self.file_name = f"db_{backbone_name}_{index_type}.index"

        if index_type == "COS":
            self.feature_extractor = feature_extractor_factory(
                backbone_name, normalize=True, device=device
            )
        else:
            self.feature_extractor = feature_extractor_factory(
                backbone_name, device=device
            )

        print(f"--------------------------------")
        print(f"backbone: {backbone_name}")
        print(f"index_type: {index_type}")
        print(f"--------------------------------")

    def init(self, image_pils):
        features = []
        for image_pil in tqdm(image_pils, desc="Extracting features"):
            cur_feature = self.feature_extractor(image_pil)
            features.append(cur_feature)
        features = torch.cat(features, axis=0)

        # for knn
        dim = features[0].size()[-1]
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            # for cos and inner product
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(features)

        print(f"Database init complete!")

    def save(self, root_path):
        os.makedirs(root_path, exist_ok=True)
        file_path = os.path.join(root_path, self.file_name)
        if not os.path.exists(file_path):
            faiss.write_index(self.index, file_path)

    def load(self, root_path):
        file_path = os.path.join(root_path, self.file_name)
        if not os.path.exists(file_path):
            # file not exists
            # load fail
            return False
        self.index = faiss.read_index(file_path)
        # load success
        return True
        # # load infos
        # self._load_infos(os.path.join(root_path, "info.pkl"))
        # print(f"load complete!")

    def query_multi(self, queries, k=1):
        pass

    def query(self, image_pil, k=1):
        feature = self.feature_extractor(image_pil)
        D, I = self.index.search(feature, k)
        idxs = I[0]

        return idxs

    def close(self, root_path=None):
        # clear cuda
        self.feature_extractor.close()
        if root_path is not None:
            self.save(root_path)


class H5DB(ImageDatabase):

    def __init__(self, backbone_name="resnet", index_type="L2", device="cuda"):
        super().__init__(backbone_name, index_type, device)
        self.info_file_name = f"info.pkl"

    def init(self, hdf5_paths=None, root_path=None):
        assert (
            hdf5_paths is not None or root_path is not None
        ), "Must give hdf5 paths or root path"
        if not (root_path is not None and self.load(root_path)):
            # only init from hdf5_path if root path is None or file not exists
            self.infos = []
            for path in tqdm(hdf5_paths, desc="Reading hdf5"):
                self.infos.append(
                    dict(path=path, image_pil=self.get_first_frame_pil(path))
                )
            self._init_base()

    def query(self, param, k=1):
        if isinstance(param, str):
            # hdf5 path
            param = self.get_first_frame_pil(param)
        idxs = super().query(param, k=k)
        result = []
        for idx in idxs:
            result.append(self.infos[idx])
        return result

    def save(self, root_path):
        super().save(root_path)

        file_path = os.path.join(root_path, self.info_file_name)
        if not os.path.exists(file_path):
            self._save_infos(file_path)

    def load(self, root_path):
        file_path = os.path.join(root_path, self.info_file_name)
        if os.path.exists(file_path):
            self._load_infos(file_path)

            state = super().load(root_path)
            if state == False:
                # if db.index not exist
                # init base class by infos
                self._init_base()
            return True
        else:
            return False

    def get_first_frame_pil(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            robot_video = f["/cam_data/robot_camera"][()]
            # BGR -> RGB
            if robot_video.shape[3] == 3:
                robot_video = robot_video[:, :, :, ::-1]

            first_frame = robot_video[0]
            first_frame = np.uint8(first_frame)
            first_frame_pil = Image.fromarray(first_frame)

        return first_frame_pil

    def _init_base(self):
        super().init([item["image_pil"] for item in self.infos])

    def _pil_image_to_bytes(self, img):
        byte_stream = io.BytesIO()
        img.save(byte_stream, format="PNG")
        return byte_stream.getvalue()

    def _bytes_to_pil_image(self, byte_data):
        byte_stream = io.BytesIO(byte_data)
        return Image.open(byte_stream)

    def _save_infos(self, save_path):
        infos = []
        for info in self.infos:
            infos.append(
                dict(
                    path=info["path"],
                    image_pil=self._pil_image_to_bytes(info["image_pil"]),
                )
            )

        with open(save_path, "wb") as f:
            pickle.dump(infos, f)

    def _load_infos(self, save_path):
        with open(save_path, "rb") as f:
            infos_raw = pickle.load(f)
        self.infos = []
        for info in infos_raw:
            self.infos.append(
                dict(
                    path=info["path"],
                    image_pil=self._bytes_to_pil_image(info["image_pil"]),
                )
            )
