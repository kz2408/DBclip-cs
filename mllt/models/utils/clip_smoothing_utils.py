import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPImageProcessor(nn.Module):
    """处理图像以满足CLIP模型的输入要求

    CLIP模型期望输入的图像：
    1. 归一化到[0, 1]
    2. 然后归一化均值(0.48145466, 0.4578275, 0.40821073)
      和标准差(0.26862954, 0.26130258, 0.27577711)
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x):
        """转换输入图像到CLIP格式

        Args:
            x: 输入图像，通常已经归一化到ImageNet均值和标准差

        Returns:
            处理后的图像，适合CLIP模型
        """
        # 反归一化到[0, 1]
        # 假设输入是按照ImageNet均值和标准差归一化的
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)

        # 反归一化
        x = x * imagenet_std + imagenet_mean

        # 裁剪到[0, 1]范围
        x = torch.clamp(x, 0, 1)

        # 应用CLIP归一化
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return x