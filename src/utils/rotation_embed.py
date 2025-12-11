import numpy as np
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod


def normalize(vector, axis=0):
    return vector / np.linalg.norm(vector, axis=axis, keepdims=True)


class RotationEmbed(ABC):
    def __init__(self):
        super().__init__()

    def angles_to_embed(self, angles):
        # angles 是一个 batch 输入，形状为 (batch_size, n)
        # 计算旋转矩阵
        matrix = self._angles_to_matrix(angles)
        # 对所有样本提取前两列并展平
        return matrix[..., :2].reshape(matrix.shape[0], -1)

    def embed_to_angles(self, embed):
        # embed 是一个 batch 输入，形状为 (batch_size, 6)
        embed = embed.reshape(-1, 3, 2)
        a1 = embed[..., 0]  # 第一列 (b, 3)
        a2 = embed[..., 1]  # 第二列 (b, 3)
        b1 = normalize(a1, axis=1)
        b2 = normalize(a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1, axis=1)
        b3 = np.cross(b1, b2)
        matrix = np.stack((b1, b2, b3), axis=-1)
        return self._matrix_to_angles(matrix)

    @abstractmethod
    def _angles_to_matrix(self, angles):
        pass

    @abstractmethod
    def _matrix_to_angles(self, matrix):
        pass


class EulerRotationEmbed(RotationEmbed):
    def __init__(self, seq="xyz", degrees=True):
        super().__init__()
        self.seq = seq
        self.degrees = degrees

    def _angles_to_matrix(self, angles):
        # angles 形状应为 (batch_size, 3)，表示每个样本的三个角度
        return R.from_euler(self.seq, angles, degrees=self.degrees).as_matrix()

    def _matrix_to_angles(self, matrix):
        # matrix 形状应为 (batch_size, 3, 3)
        return R.from_matrix(matrix).as_euler(self.seq, degrees=self.degrees)
