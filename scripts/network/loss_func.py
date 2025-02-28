"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# Description: Define the loss function for training.
"""
import torch

def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    speed_0_4 = pts_loss[speed < 0.4].mean()
    speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    speed_1_0 = pts_loss[speed > 1.0].mean()
    if ~speed_1_0.isnan():
        weight_loss += speed_1_0
    if ~speed_0_4.isnan():
        weight_loss += speed_0_4
    if ~speed_mid.isnan():
        weight_loss += speed_mid
    return weight_loss



def restoreLoss(restore_per_point):

    loss = restore_per_point.mean()
    # pred = res_dict['est_flow']
    # gt = res_dict['gt_flow']

    # mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    # pred = pred[mask_no_nan].reshape(-1, 3)
    # gt = gt[mask_no_nan].reshape(-1, 3)

    # speed = gt.norm(dim=1, p=2) / 0.1
    # # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    # pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    # weight_loss = 0.0
    # speed_0_4 = pts_loss[speed < 0.4].mean()
    # speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    # speed_1_0 = pts_loss[speed > 1.0].mean()
    # if ~speed_1_0.isnan():
    #     weight_loss += speed_1_0
    # if ~speed_0_4.isnan():
    #     weight_loss += speed_0_4
    # if ~speed_mid.isnan():
    #     weight_loss += speed_mid
    return loss

# ref from zeroflow loss class FastFlow3DDistillationLoss()
def zeroflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    # gt_speed = torch.norm(gt, dim=1, p=2) * 10.0
    gt_speed = torch.linalg.vector_norm(gt, dim=-1) * 10.0
    
    mins = torch.ones_like(gt_speed) * 0.1
    maxs = torch.ones_like(gt_speed)
    importance_scale = torch.max(mins, torch.min(1.8 * gt_speed - 0.8, maxs))
    # error = torch.norm(pred - gt, dim=1, p=2) * importance_scale
    error = error * importance_scale
    return error.mean()

# ref from zeroflow loss class FastFlow3DSupervisedLoss()
def ff3dLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    # error = torch.norm(pred - gt, dim=1, p=2)
    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    is_foreground_class = (classes > 0) # 0 is background, ref: FOREGROUND_BACKGROUND_BREAKDOWN
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return error.mean()


# add feature loss
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptionLoss3D(nn.Module):
    def __init__(self, in_channels=16, target_channels=3, feature_layer="conv4_2", loss_type="l1"):
        """
        计算五维张量 (B, C, H, W, Z) 的 VGG 感知损失
        :param in_channels: 原始特征通道数（如 16）
        :param target_channels: VGG 需要的输入通道数（默认 3）
        :param feature_layer: 选择 VGG 的特征提取层
        :param loss_type: "l1" 或 "l2"
        """
        super(VGGPerceptionLoss3D, self).__init__()
        
        # 1×1 卷积将 C 维度降到 3
        self.channel_mapper = nn.Conv2d(in_channels, target_channels, kernel_size=1)

        # 载入 VGG 并选择特征提取层
        vgg = models.vgg16(pretrained=True).features
        if feature_layer == "conv4_2":
            self.feature_extractor = nn.Sequential(*vgg[:23])  # 取 VGG16 的 `conv4_2`
        elif feature_layer == "conv5_2":
            self.feature_extractor = nn.Sequential(*vgg[:30])  # 取 VGG16 的 `conv5_2`
        else:
            raise ValueError("feature_layer must be 'conv4_2' or 'conv5_2'")

        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # 冻结 VGG 参数

        # 选择损失函数
        self.loss_fn = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()

    def forward(self, pred, gt):
        B, C, H, W, Z = pred.shape  # 解析 5D 形状
        loss = 0

        for z in range(Z):  # 遍历 Z 维度的每一层
            pred_slice = pred[:, :, :, :, z]  # 取 z 轴上的 2D 切片 (B, C, H, W)
            gt_slice = gt[:, :, :, :, z]

            # 使用 1×1 卷积降维到 3 通道
            pred_slice = self.channel_mapper(pred_slice)
            gt_slice = self.channel_mapper(gt_slice)

            # 计算 VGG 感知损失
            loss += self.loss_fn(self.feature_extractor(pred_slice), self.feature_extractor(gt_slice))

        return loss / Z  # 取平均，确保所有 Z 层贡献均衡

# 示例
B, C, H, W, Z = 4, 16, 224, 224, 10  # 五维体数据
pred = torch.randn(B, C, H, W, Z)
gt = torch.randn(B, C, H, W, Z)

# 计算 VGG 3D 感知损失
loss_fn = VGGPerceptionLoss3D(in_channels=16)
loss = loss_fn(pred, gt)
print("VGG 3D Perception Loss:", loss.item())
