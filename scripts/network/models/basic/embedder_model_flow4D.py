from numpy import dtype
import torch
import torch.nn as nn

from .make_voxels import HardVoxelizer, DynamicVoxelizer
from .process_voxels import PillarFeatureNet, DynamicPillarFeatureNet, DynamicPillarFeatureNet_flow4D
from .scatter import PointPillarsScatter

import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True
import spconv.pytorch as spconv
from .generator import VarianceSchedule
from .network_4D import Network_3D, Seperate_to_pc0_pc1
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict
import faiss
import time

class HardEmbedder(nn.Module):

    def __init__(self,
                 voxel_size=(0.2, 0.2, 4),
                 pseudo_image_dims=(350, 350),
                 point_cloud_range=(-35, -35, -3, 35, 35, 1),
                 max_points_per_voxel=128,
                 feat_channels=64) -> None:
        super().__init__()
        self.voxelizer = HardVoxelizer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel)
        self.feature_net = PillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"

        output_voxels, output_voxel_coords, points_per_voxel = self.voxelizer(
            points)
        output_features = self.feature_net(output_voxels, points_per_voxel,
                                           output_voxel_coords)
        pseudoimage = self.scatter(output_features, output_voxel_coords)

        return pseudoimage


class DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            voxel_feats, voxel_coors = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list


class DynamicEmbedder_4D(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict, training_flag) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']
        # if training_flag:
        #     frame_keys += ['pc1s_gt']
        pc0_point_feats_lst = []
        voxel_feats_list_batch_dict = {}
        voxel_coors_list_batch_dict = {}

        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates) #经过batchnorm_1d

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)
                
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1) #!这里之后都是xyz的顺序了

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4


            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)
            if frame_key == 'pc1s' or frame_key == 'pc1s_gt':
                voxel_feats_list_batch_dict[frame_key] = voxel_feats_list_batch
                voxel_coors_list_batch_dict[frame_key] = coors_batch_sp_4d 
            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp_4d)  #batch_idx, x,y,z, time_idx 

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]
                pc0s_voxel_feature = voxel_feats_sp

            # if frame_key == 'pc1s_gt':
            #     pc1s_3dvoxel_infos_lst = voxel_info_list


        all_voxel_feats_sp = torch.cat(voxel_feats_list[:2], dim=0) #
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list[:2], dim=0)
        # print("self.voxel_spatial_shape", self.voxel_spatial_shape) [512, 512, 32, 2]
        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), self.voxel_spatial_shape, int(batch_index + 1))

        output = {
            '4d_tensor': sparse_tensor_4d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_mum_voxels': pc0s_num_voxels, # 第0帧包括所有batch的voxel数量
        }
        # if training_flag:
        output['voxel_feats_list_batch_dict'] = voxel_feats_list_batch_dict
        output['voxel_coors_list_batch_dict'] = voxel_coors_list_batch_dict
        output['pc0s_voxel_feature'] = pc0s_voxel_feature

        return output


class DynamicEmbedder_4D_less_to_more(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s_all', 'pc1s_all']
        # if training_flag:
        #     frame_keys += ['pc0_all', 'pc1_all']
        pc0_point_feats_lst = []
        voxel_feats_list_batch_dict = {}
        voxel_coors_list_batch_dict = {}

        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates) #经过batchnorm_1d

                # if frame_key == 'pc0s':
                #     pc0_point_feats_lst.append(point_feats)
                
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1) #!这里之后都是xyz的顺序了

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4


            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)
            # if frame_key == 'pc1s' or frame_key == 'pc1s_gt':
            voxel_feats_list_batch_dict[frame_key] = voxel_feats_list_batch
            voxel_coors_list_batch_dict[frame_key] = coors_batch_sp_4d
            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp_4d)  #batch_idx, x,y,z, time_idx 

            # if frame_key == 'pc0s':
            #     pc0s_3dvoxel_infos_lst = voxel_info_list
            #     pc0s_num_voxels = voxel_feats_sp.shape[0]
            #     pc0s_voxel_feature = voxel_feats_sp

            # if frame_key == 'pc1s_gt':
            #     pc1s_3dvoxel_infos_lst = voxel_info_list


        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0) #! 稀疏pc0, pc1的特征拼接
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list, dim=0)
        # print("self.voxel_spatial_shape", self.voxel_spatial_shape) [512, 512, 32, 2]
        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), self.voxel_spatial_shape, int(batch_index + 1))

        output = {
            '4d_tensor': sparse_tensor_4d,
            # 'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            # 'pc0_point_feats_lst': pc0_point_feats_lst,
            # 'pc0_mum_voxels': pc0s_num_voxels, # 第0帧包括所有batch的voxel数量
        }
        # if training_flag:
        output['voxel_feats_list_batch_dict'] = voxel_feats_list_batch_dict
        output['voxel_coors_list_batch_dict'] = voxel_coors_list_batch_dict
        # output['pc0s_voxel_feature'] = pc0s_voxel_feature

        return output


#less only one frame
class DynamicEmbedder_4D_less(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']
        # if training_flag:
        #     frame_keys += ['pc0_all', 'pc1_all']
        pc0_point_feats_lst = []
        voxel_feats_list_batch_dict = {}
        voxel_coors_list_batch_dict = {}

        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates) #经过batchnorm_1d

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)
                
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1) #!这里之后都是xyz的顺序了

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4


            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)
            # if frame_key == 'pc1s' or frame_key == 'pc1s_gt':
            voxel_feats_list_batch_dict[frame_key] = voxel_feats_list_batch
            voxel_coors_list_batch_dict[frame_key] = coors_batch_sp_4d
            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp_4d)  #batch_idx, x,y,z, time_idx 

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]
                pc0s_voxel_feature = voxel_feats_sp

            # if frame_key == 'pc1s_gt':
            #     pc1s_3dvoxel_infos_lst = voxel_info_list


        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0) #! 稀疏pc0, pc1的特征拼接
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list, dim=0)
        # print("self.voxel_spatial_shape", self.voxel_spatial_shape) [512, 512, 32, 2]
        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), self.voxel_spatial_shape, int(batch_index + 1))

        output = {
            '4d_tensor': sparse_tensor_4d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_mum_voxels': pc0s_num_voxels, # 第0帧包括所有batch的voxel数量
        }
        # if training_flag:
        output['voxel_feats_list_batch_dict'] = voxel_feats_list_batch_dict
        output['voxel_coors_list_batch_dict'] = voxel_coors_list_batch_dict
        output['pc0s_voxel_feature'] = pc0s_voxel_feature

        return output

# add point noise
class DynamicEmbedder_4D_less_to_more_add_noise(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims
        self.var_sched = VarianceSchedule()
        self.point_cloud_range = point_cloud_range

    def forward(self, input_dict, training_flag) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        voxel_coors_list_4D = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']
        batch_sizes = input_dict['pc0s'].shape[0]
        device = input_dict['pc0s'].device

        add_noise_frame_keys = ['pc0_all', 'pc1_all']

        if training_flag:
            #!pc0和pc1加同样的噪声强度
            ts = self.var_sched.recurrent_uniform_sampling(batch_sizes, 1)
            ts = ts[0].to(device)
        else:
            ts = torch.full((batch_sizes,), 1000.0)
            ts = ts.to(device)

        # time_code = self.var_sched.time_mlp(ts)

        pc0_point_feats_lst = []
        voxel_feats_list_batch_dict = {}
        voxel_coors_list_batch_dict = {}
        min_vals = torch.tensor(self.point_cloud_range[:3], device=device)
        max_vals = torch.tensor(self.point_cloud_range[3:], device=device)
        
        #! point add noise
        add_noise_dict = {}
        for t_index, add_noise_frame_key in enumerate(add_noise_frame_keys):
            add_noise_dict[add_noise_frame_key] = {}
            if training_flag:
                pc_ori = input_dict[add_noise_frame_key]
                voxel_info_list = self.voxelizer(pc_ori) 
                pc_xt_list = []
                noise_list = [] 
                
                 #! valid point
                for b_idx in range(batch_sizes):
                    point_i = voxel_info_list[b_idx]['points']
                    # 归一化
                    point_normalized = (point_i - min_vals) / (max_vals - min_vals) * 2 - 1 # n*3 
                    noise =  torch.randn_like(point_normalized, device= device).float() 
                    pc_xt = self.var_sched.q_sample(x_start=point_normalized, t=ts[b_idx], noise = noise)
                    pc_xt = torch.clamp(pc_xt, min=-1, max=1) 
                    pc_xt_list.append(pc_xt)
                    noise_list.append(noise)

                add_noise_dict[add_noise_frame_key]['pc_xt_list'] = pc_xt_list
                add_noise_dict[add_noise_frame_key]['noise_list'] = noise_list
                
            else:
                if add_noise_frame_key == 'pc0_all':
                    pc_noise=  torch.randn((batch_sizes, 4*input_dict['pc0s'].shape[1],3), device= device).float()
                elif add_noise_frame_key == 'pc1_all':
                    pc_noise=  torch.randn((batch_sizes, 4*input_dict['pc1s'].shape[1],3), device= device).float()
                add_noise_dict[add_noise_frame_key]['pc_xt_list'] = pc_noise
            add_noise_dict[add_noise_frame_key]['t'] = ts

                
        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates) #经过batchnorm_1d

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)
                
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                if frame_key == 'pc0s':
                    voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1) #!这里之后都是xyz的顺序了
                else:
                    voxel_coors_batch = torch.cat([batch_indices + batch_sizes, voxel_coors[:, [2, 1, 0]]], dim=1) #! batch上叠加两帧

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4


            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)
            # if frame_key == 'pc1s' or frame_key == 'pc1s_gt':
            voxel_feats_list_batch_dict[frame_key] = voxel_feats_list_batch
            voxel_coors_list_batch_dict[frame_key] = coors_batch_sp
            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp)  #batch_idx, x,y,z, time_idx 
            voxel_coors_list_4D.append(coors_batch_sp_4d)  #batch_idx, x,y,z, time_idx 

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]
                pc0s_voxel_feature = voxel_feats_sp

            # if frame_key == 'pc1s_gt':
            #     pc1s_3dvoxel_infos_lst = voxel_info_list


        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0) #! 稀疏pc0, pc1的特征拼接
        all_coors_batch_sp_3d = torch.cat(voxel_coors_list, dim=0)
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list_4D, dim=0)

        # print("self.voxel_spatial_shape", self.voxel_spatial_shape) [512, 512, 32, 2]
        sparse_tensor_3d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_3d.contiguous(), \
                                                   self.voxel_spatial_shape[:3], int(batch_index + 1))
        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), \
                                                   self.voxel_spatial_shape, int(batch_index + 1))
        

        output = {
            '4d_tensor': sparse_tensor_4d,
            '3d_tensor': sparse_tensor_3d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_mum_voxels': pc0s_num_voxels, # 第0帧包括所有batch的voxel数量
        }
        # if training_flag:
        output['voxel_feats_list_batch_dict'] = voxel_feats_list_batch_dict
        output['voxel_coors_list_batch_dict'] = voxel_coors_list_batch_dict
        output['pc0s_voxel_feature'] = pc0s_voxel_feature
        output['add_noise_dict'] = add_noise_dict

        return output
    

class PCNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super(PCNet, self).__init__()
        self.fea_layer = nn.Linear(dim_in, dim_out)
        self.cond_bias = nn.Linear(dim_cond, dim_out, bias=False)
        self.cond_gate = nn.Linear(dim_cond, dim_out)

    def forward(self, fea, cond):
        gate = torch.sigmoid(self.cond_gate(cond))
        bias = self.cond_bias(cond)
        out = self.fea_layer(fea) * gate + bias
        return out



# add point noise
class DynamicEmbedder_4D_offset_add_noise(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        # self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
        #                                   point_cloud_range=point_cloud_range)
        voxel_size_1_2 = [v/2 for v in voxel_size]
        voxel_size_4 = [v*4 for v in voxel_size]
        self.voxelizer_1_2 = DynamicVoxelizer(voxel_size=voxel_size_1_2,
                                    point_cloud_range=point_cloud_range)
        self.voxelizer_4 = DynamicVoxelizer(voxel_size=voxel_size_4,
                            point_cloud_range=point_cloud_range)
        
        self.feature_net_1_2 = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size_1_2, #!
            mode='avg')
        
        # self.feature_net = DynamicPillarFeatureNet_flow4D(
        #     in_channels=3,
        #     feat_channels=(feat_channels, ),
        #     point_cloud_range=point_cloud_range,
        #     voxel_size=voxel_size, #!
        #     mode='avg')
        
        point_output_ch = 16
        voxel_output_ch = 16
        self.network_3D = Network_3D(in_channel=point_output_ch, out_channel=voxel_output_ch)
        # self.scatter = PointPillarsScatter(in_channels=feat_channels,
        #                                    output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims
        self.voxel_spatial_shape_1_2 = [int(v*2) for v in self.voxel_spatial_shape]
        self.voxel_spatial_shape_4 = [int(v/4) for v in self.voxel_spatial_shape]
        self.var_sched = VarianceSchedule()
        self.point_cloud_range = point_cloud_range
        self.seperate_to_pc0_pc1 = Seperate_to_pc0_pc1()

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        #! larger voxel
        self.vx = voxel_size_4[0]
        self.vy = voxel_size_4[1]
        self.vz = voxel_size_4[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.voxel_size = voxel_size
        self.zeromask = nn.Parameter(torch.zeros(64))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 32),
        )
        cond_dims = 64+32+4
        self.layers = nn.ModuleList([
            PCNet(3, 64, cond_dims),
            PCNet(64, 128, cond_dims),
            PCNet(128, 64, cond_dims),
            PCNet(64, 3, cond_dims)
        ])
        self.act = F.leaky_relu


    def offset_voxel_center(self,features, coors, dtype = torch.float32): 
        voxel_center_xyz = torch.stack([
            coors[:, 2].to(dtype) * self.vx + self.x_offset,
            coors[:, 1].to(dtype) * self.vy + self.y_offset,
            coors[:, 0].to(dtype) * self.vz + self.z_offset
            ], dim=1)
        if features is None:
            return voxel_center_xyz

        f_center = features.new_zeros(size=(features.size(0), 3))
        f_center[:, 0] = features[:, 0] - (
            coors[:, 2].type_as(features) * self.vx + self.x_offset)
        f_center[:, 1] = features[:, 1] - (
            coors[:, 1].type_as(features) * self.vy + self.y_offset)
        f_center[:, 2] = features[:, 2] - (
            coors[:, 0].type_as(features) * self.vz + self.z_offset)
        
 

        return f_center, voxel_center_xyz
    






    def fast_knn_gpu_before(self, sparse_points, sparse_voxels, dense_points, dense_voxels, K):
        """
        以 Voxel 为基准，使用 PyTorch GPU 进行高效 KNN 搜索，每个稀疏点找到同 voxel 内最近的 K 个稠密点。

        Args:
            sparse_points (torch.Tensor): (N, 3) 稀疏点坐标
            sparse_voxels (torch.Tensor): (N, 3) 稀疏点的 voxel 坐标
            dense_points (torch.Tensor): (M, 3) 稠密点坐标
            dense_voxels (torch.Tensor): (M, 3) 稠密点的 voxel 坐标
            K (int): 需要寻找的最近邻个数

        Returns:
            torch.Tensor: (N, K, 3) 形状的 Tensor，存放在 GPU 上，按 `sparse_points` 原顺序排列。
        """
        device = sparse_points.device
        N = sparse_points.shape[0]

        # 1. 计算每个 voxel 的唯一 key
        sparse_voxel_keys = (sparse_voxels[:, 0] * self.voxel_spatial_shape_4[0] * self.voxel_spatial_shape_4[1] +
                            sparse_voxels[:, 1] * self.voxel_spatial_shape_4[0] + sparse_voxels[:, 2])
        
        dense_voxel_keys = (dense_voxels[:, 0] * self.voxel_spatial_shape_4[0] * self.voxel_spatial_shape_4[1] +
                            dense_voxels[:, 1] * self.voxel_spatial_shape_4[0] + dense_voxels[:, 2])

        # 2. 在 CPU 端构建 Voxel 内点的映射
        voxel_to_dense = defaultdict(list)
        for i, key in enumerate(dense_voxel_keys.cpu().tolist()):
            voxel_to_dense[key].append(i)

        # 3. **提前分配 Tensor**
        results = torch.full((N, K, 3), float('nan'), device=device)  # 预填充 NaN 以检查未赋值的点
        mask = torch.zeros(N, dtype=torch.bool, device=device)  # 记录哪些点没有找到足够的邻居

        # 4. **利用 `inverse_indices` 进行索引匹配**
        unique_sparse_keys, inverse_indices = torch.unique(sparse_voxel_keys, return_inverse=True)

        # 向量化处理所有 voxel
        for i, voxel_key in enumerate(unique_sparse_keys):
            sparse_indices = torch.where(inverse_indices == i)[0]  # 直接用 inverse_indices 查找
            sparse_pts = sparse_points[sparse_indices]  # (Nsparse, 3)

            if voxel_key.item() in voxel_to_dense:
                dense_indices = voxel_to_dense[voxel_key.item()]
                dense_pts = dense_points[dense_indices]  # (Ndense, 3)

                # **并行计算 KNN**
                distances = torch.cdist(sparse_pts, dense_pts)  # (Nsparse, Ndense)
                knn_indices = distances.topk(k=min(K, len(dense_pts)), dim=1, largest=False).indices

                # **填充 K 个邻居**
                num_neighbors = knn_indices.shape[1]
                if num_neighbors < K:
                    # **从已有的 topk 结果里随机重复采样**
                    repeat_indices = torch.randint(0, num_neighbors, (K - num_neighbors,), device=device)
                    sampled_knn_indices = torch.cat([knn_indices, knn_indices[:, repeat_indices]], dim=1)
                    mask[sparse_indices] = True  # 标记哪些点进行了补全
                else:
                    sampled_knn_indices = knn_indices  # 直接使用已有的 K 个

                # **存储结果（使用 scatter_ 直接存入原顺序）**
                results[sparse_indices] = dense_pts[sampled_knn_indices]

        return results  # 直接返回 CUDA 上的 Tensor，保证顺序











    def fast_knn_gpu(self, sparse_points, sparse_voxels, dense_points, dense_voxels, K):
        """
        以 Voxel 为基准，使用 PyTorch GPU 进行高效 KNN 搜索，每个稀疏点找到同 voxel 内最近的 K 个稠密点。

        Args:
            sparse_points (torch.Tensor): (N, 3) 稀疏点坐标 (float32, CUDA)
            sparse_voxels (torch.Tensor): (N, 3) 稀疏点的 voxel 坐标 (int32, CUDA)
            dense_points (torch.Tensor): (M, 3) 稠密点坐标 (float32, CUDA)
            dense_voxels (torch.Tensor): (M, 3) 稠密点的 voxel 坐标 (int32, CUDA)
            K (int): 需要寻找的最近邻个数

        Returns:
            torch.Tensor: (N, K, 3) 形状的 Tensor存放在 GPU 上，按 `sparse_points` 原顺序排列。
        """
        device = sparse_points.device
        N, M = sparse_points.shape[0], dense_points.shape[0]

        # 1. 计算 Voxel 索引，避免 CPU 端字典操作
        voxel_spatial_shape = self.voxel_spatial_shape_4  # 假设是 [x, y, z] 大小
        sparse_voxel_keys = (sparse_voxels[:, 0] * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
                            sparse_voxels[:, 1] * voxel_spatial_shape[0] + sparse_voxels[:, 2])
        dense_voxel_keys = (dense_voxels[:, 0] * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
                            dense_voxels[:, 1] * voxel_spatial_shape[0] + dense_voxels[:, 2])

        # 2. 获取唯一的 sparse_voxel_keys 和其索引
        unique_sparse_keys, inverse_indices = torch.unique(sparse_voxel_keys, return_inverse=True)

        # 3. **利用 Tensor 直接构建索引映射**
        dense_sorted_indices = torch.argsort(dense_voxel_keys)  # 先排序
        sorted_dense_keys = dense_voxel_keys[dense_sorted_indices]  # 排序后的 voxel keys

        # 4. 在 GPU 端搜索 Voxel 中的点索引，避免 Python `defaultdict`
        voxel_start_idx = torch.searchsorted(sorted_dense_keys, unique_sparse_keys)  # 查找起始索引
        voxel_end_idx = torch.searchsorted(sorted_dense_keys, unique_sparse_keys, side="right")  # 查找结束索引

        # 5. **提前分配 `(N, K, 3)` Tensor**
        results = torch.full((N, K, 3), float('nan'), device=device)  # 直接创建 CUDA Tensor

        # 6. **完全并行 KNN 计算**
        for i, voxel_key in enumerate(unique_sparse_keys):
            sparse_indices = torch.where(inverse_indices == i)[0]  # 取出该 voxel 里的所有 sparse 点
            sparse_pts = sparse_points[sparse_indices]  # (Nsparse, 3)

            start_idx, end_idx = voxel_start_idx[i], voxel_end_idx[i]
            if start_idx < end_idx:  # 该 voxel 里有稠密点
                dense_indices = dense_sorted_indices[start_idx:end_idx]  # 提取该 voxel 内的 dense 索引
                dense_pts = dense_points[dense_indices]  # (Ndense, 3)

                # **计算 KNN**
                distances = torch.cdist(sparse_pts, dense_pts)  # (Nsparse, Ndense)
                knn_indices = distances.topk(k=min(K, len(dense_pts)), dim=1, largest=False).indices  # 取最近 K 个

                # **填充 K 个邻居**
                num_neighbors = knn_indices.shape[1]
                if num_neighbors < K:
                    repeat_indices = torch.randint(0, num_neighbors, (K - num_neighbors,), device=device)
                    knn_indices = torch.cat([knn_indices, knn_indices[:, repeat_indices]], dim=1)

                # **并行存储到 `results`** 
                #! 这里的 `results` 已经是按 `sparse_points` 原顺序排列的
                results[sparse_indices] = dense_pts[knn_indices]

        return results  # 直接返回 CUDA Tensor


    def fast_knn_faiss(self, sparse_points, sparse_voxels, dense_points, dense_voxels, K):
        """
        以 Voxel 为基准，使用 FAISS 进行 GPU 加速 KNN 搜索，每个稀疏点找到同 voxel 内最近的 K 个稠密点。

        Args:
            sparse_points (torch.Tensor): (N, 3) 稀疏点坐标 (float32, CUDA)
            sparse_voxels (torch.Tensor): (N, 3) 稀疏点的 voxel 坐标 (int32, CUDA)
            dense_points (torch.Tensor): (M, 3) 稠密点坐标 (float32, CUDA)
            dense_voxels (torch.Tensor): (M, 3) 稠密点的 voxel 坐标 (int32, CUDA)
            K (int): 需要寻找的最近邻个数

        Returns:
            torch.Tensor: (N, K, 3) 形状的 Tensor，存放在 GPU 上，按 `sparse_points` 原顺序排列。
        """
        device = sparse_points.device
        N, M = sparse_points.shape[0], dense_points.shape[0]

        # 1. 计算 Voxel 索引
        voxel_spatial_shape = self.voxel_spatial_shape_4
        sparse_voxel_keys = (sparse_voxels[:, 0] * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
                            sparse_voxels[:, 1] * voxel_spatial_shape[0] + sparse_voxels[:, 2])
        dense_voxel_keys = (dense_voxels[:, 0] * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
                            dense_voxels[:, 1] * voxel_spatial_shape[0] + dense_voxels[:, 2])

        # 2. 获取唯一的 sparse_voxel_keys 和其索引
        unique_sparse_keys, inverse_indices = torch.unique(sparse_voxel_keys, return_inverse=True)

        # 3. **使用 FAISS 进行 GPU 加速 KNN 搜索**
        index = faiss.IndexFlatL2(3)  # L2 距离索引 (欧几里得距离)
        res = faiss.StandardGpuResources()  # GPU 资源管理
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引移动到 GPU

        # 4. **构建 GPU KNN 索引**
        gpu_index.add(dense_points.cpu().numpy())  # FAISS 需要 NumPy 输入

        # 5. **执行 KNN 搜索**
        distances, knn_indices = gpu_index.search(sparse_points.cpu().numpy(), K)  # 在 GPU 上搜索最近 K 个点

        # 6. **转换为 PyTorch Tensor 并返回**
        knn_indices = torch.tensor(knn_indices, dtype=torch.long, device=device)  # (N, K)
        results = dense_points[knn_indices]  # (N, K, 3)

        return results  # 直接返回 CUDA Tensor



    def fast_knn_faiss_hnsw(self, sparse_points, sparse_voxels, dense_points, dense_voxels, K):
        """
        以 Voxel 为基准，使用 FAISS + HNSW 进行 **近似最近邻 (ANN)** 搜索，加速 KNN。

        Args:
            sparse_points (torch.Tensor): (N, 3) 稀疏点坐标 (float32, CUDA)
            sparse_voxels (torch.Tensor): (N, 3) 稀疏点的 voxel 坐标 (int32, CUDA)
            dense_points (torch.Tensor): (M, 3) 稠密点坐标 (float32, CUDA)
            dense_voxels (torch.Tensor): (M, 3) 稠密点的 voxel 坐标 (int32, CUDA)
            K (int): 需要寻找的最近邻个数

        Returns:
            torch.Tensor: (N, K, 3) 形状的 Tensor，存放在 GPU 上，按 `sparse_points` 原顺序排列。
        """
        device = sparse_points.device
        N, M = sparse_points.shape[0], dense_points.shape[0]

        # 1. **使用 FAISS HNSW 近似最近邻搜索**
        index = faiss.IndexHNSWFlat(3, 32)  # 使用 HNSW (32 近邻)，比 `IndexFlatL2` 更快
        res = faiss.StandardGpuResources()  # GPU 资源管理
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引移动到 GPU

        # 2. **构建 GPU HNSW 索引**
        gpu_index.add(dense_points.cpu().numpy())  # FAISS 需要 NumPy 输入

        # 3. **执行 KNN 搜索**
        distances, knn_indices = gpu_index.search(sparse_points.cpu().numpy(), K)  # 近似最近邻搜索

        # 4. **转换为 PyTorch Tensor 并返回**
        knn_indices = torch.tensor(knn_indices, dtype=torch.long, device=device)  # (N, K)
        results = dense_points[knn_indices]  # (N, K, 3)

        return results  # 直接返回 CUDA Tensor
    
    def forward(self, input_dict, training_flag) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        voxel_coors_list_4D = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']
        batch_sizes = input_dict['pc0s'].shape[0]
        device = input_dict['pc0s'].device
        
        #! generate sparse condition feature
        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer_1_2(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net_1_2(points, coordinates) #经过batchnorm_1d

                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                if frame_key == 'pc0s':
                    voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1) #!这里之后都是xyz的顺序了
                else:
                    voxel_coors_batch = torch.cat([batch_indices + batch_sizes, voxel_coors[:, [2, 1, 0]]], dim=1) #! batch上叠加两帧

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4

            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp)  #batch_idx, x,y,z


        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0) #! 稀疏pc0, pc1的特征拼接
        all_coors_batch_sp_3d = torch.cat(voxel_coors_list, dim=0)

        # print("self.voxel_spatial_shape", self.voxel_spatial_shape) [512, 512, 32, 2]
        sparse_tensor_3d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_3d.contiguous(), \
                                                   self.voxel_spatial_shape_1_2[:3], int((batch_index + 1)*2))
        condition = self.network_3D(sparse_tensor_3d) # N, 16 spatial_shape[128,128,8]  0.2 voxel的1/4
        pc0_condition, pc1_condition = self.seperate_to_pc0_pc1(condition, batch_sizes) # N, 16 spatial_shape[128,128,8]  0.2 voxel的1/4
        total_loss = 0.0
        # #! point add noise

        add_noise_frame_keys = ['pc0_all', 'pc1_all']

        if training_flag:
            #!pc0和pc1加同样的噪声强度
            total_loss = []
            ts = self.var_sched.recurrent_uniform_sampling(batch_sizes, 1)
            ts = ts[0].to(device)
        else:
            ts = torch.full((batch_sizes,), int(999))
            ts = ts.to(device)

        time_code = self.var_sched.time_mlp(ts)

        
        restore_point_dict = {}

        for time_i, noise_frame_key in enumerate(add_noise_frame_keys):
            
            if training_flag:
                if noise_frame_key == 'pc0_all':
                    point_condition = pc0_condition.dense().permute(0,2,3,4,1) #b,x,y,z,c
                    sparse_frame_key = 'pc0s'
                elif noise_frame_key == 'pc1_all':
                    point_condition = pc1_condition.dense().permute(0,2,3,4,1) #b,x,y,z,c
                    sparse_frame_key = 'pc1s'
                
                pc_all = input_dict[noise_frame_key]
                voxel_info_list_all = self.voxelizer_4(pc_all)
                pc_sparse = input_dict[sparse_frame_key]
                voxel_info_list_all_sparse = self.voxelizer_4(pc_sparse)
                # result_dict = {
                #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
                #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
                #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
                #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
                # }
                restore_point_list = []
                for b_idx, voxel_info_dict_all in enumerate(voxel_info_list_all):
                    points_all = voxel_info_dict_all['points']
                    coordinates_all = voxel_info_dict_all['voxel_coords'] #! z,y,x
                    points_sparse = voxel_info_list_all_sparse[b_idx]['points']
                    coordinates_sparse = voxel_info_list_all_sparse[b_idx]['voxel_coords'] #! z,y,x
                    # start_time1 = time.time()
                    neighbor = self.fast_knn_gpu(points_sparse, coordinates_sparse, points_all, coordinates_all, 5)
                    assert torch.isnan(neighbor).any() == False
                    # print(f"time for knn search: {time.time() - start_time1}")
                    # start_time2 = time.time()
                    # neighbor = self.fast_knn_gpu_before(points_sparse, coordinates_sparse, points_all, coordinates_all, 5)
                    # print(f"time for before: {time.time() - start_time2}")
                    # start_time3 = time.time()
                    # neighbor = self.fast_knn_faiss_hnsw(points_sparse, coordinates_sparse, points_all, coordinates_all, 5)
                    # print(f"time for faiss_hnsw: {time.time() - start_time3}")
                    
                    batch_indices_all = torch.full((coordinates_all.size(0), 1), b_idx, dtype=torch.long, device=voxel_coors.device)
                    voxel_coors_b = torch.cat([batch_indices_all, coordinates_all[:, [2, 1, 0]]], dim=1) 
                    #! voxel中心
                    points_offset_voxel_center, voxel_center_xyz = self.offset_voxel_center(points_all, coordinates_all)
                    cond_voxel_to_point = point_condition[voxel_coors_b[:, 0], voxel_coors_b[:, 1], voxel_coors_b[:, 2], voxel_coors_b[:, 3], :]
                    mask_para = ~(cond_voxel_to_point.any(dim =1))
                    cond_voxel_to_point = cond_voxel_to_point * ~mask_para[:, None] + mask_para[:, None] * self.zeromask # N, 64

                    #! voxel位置编码
                    pos_embed = self.pos_embed(voxel_center_xyz) # N, 32
                    time_embed = time_code[b_idx].repeat(cond_voxel_to_point.shape[0], 1) # N, 4
                    #! 全部条件
                    cond_all_in = torch.cat([cond_voxel_to_point, pos_embed, time_embed], dim=1) # N, 68
                    #! 点云offset加噪声
                    noise =  torch.randn_like(points_offset_voxel_center, device= device).float()
                    noise = noise * (self.vx/2) #0.4
                    out = self.var_sched.q_sample(x_start=points_offset_voxel_center, t=ts[b_idx], noise = noise) 
                    #! 确认out的范围 
                    check_out_range = False
                    if check_out_range:
                        abnormal_mask = (torch.abs(out) > 0.4).any(dim=1)
                        abnormal_num = abnormal_mask.sum().item()
                        percent = abnormal_num / points_offset_voxel_center.shape[0]
                        print(f"abnormal_num: {abnormal_num}, percent: {percent}")
                    xt = out
                    for i, layer in enumerate(self.layers):
                        out = layer(fea=out, cond=cond_all_in)
                        if i < len(self.layers) - 1:
                            out = self.act(out)
                    pred_out = out
                    # restore_loss_per_point = F.mse_loss(pred_out, points_offset_voxel_center, reduction='none') #！ change for 
                    restore_loss_per_point = torch.linalg.vector_norm(pred_out - points_offset_voxel_center, dim=-1)
                    total_loss.append(restore_loss_per_point)
                    # pred_out = self.var_sched.predict_start_from_noise(x_t = xt, t=ts[b_idx], noise = pred_noise)  # self, x_t, t, noise)
                    
                    restore_point = torch.clamp(pred_out, min = -1.0*self.vx/2, max = 1.0*self.vx/2) + voxel_center_xyz
                    restore_point_list.append(restore_point)
                    save_data = False
                    if save_data:
                        #! save data for visualization
                        restore_point_numpy = restore_point.clone().detach().cpu().numpy()
                        xt_numpy = (xt+ voxel_center_xyz).clone().detach().cpu().numpy()
                        gt_point_numpy = points_all.clone().detach().cpu().numpy()
                        data_to_save = {
                                    "restore_point": restore_point_numpy,
                                    "xt": xt_numpy,
                                    "gt_point": gt_point_numpy,
                                    'ts': ts[b_idx].item(),
                                    "sparse_point": input_dict[sparse_frame_key][b_idx].clone().detach().cpu().numpy(),
                                }
                        time_stamps = input_dict['timestamps'][b_idx]
                        scene_id = input_dict['scene_ids'][b_idx]
                        save_path = os.path.join(os.path.join(os.getcwd(), "data_to_save"), scene_id, time_stamps)
                        os.makedirs(save_path, exist_ok=True)
                        np.save(f"{save_path}/{noise_frame_key}.npy", data_to_save)
                        print(f"save data to {save_path}/{noise_frame_key}.npy")
                        file_name_path = os.path.join(os.path.join(os.getcwd(), "data_to_save")) 
                        file_name = f"{file_name_path}/all.json"
                        # 保存当前 scene 的 time_stamp 列表到文件中
                        with open(file_name, "a") as f:
                            # 将字典转换成 JSON 字符串，并写入文件，末尾添加换行
                            f.write(json.dumps(save_path) + "\n")


                restore_point_dict[noise_frame_key] = restore_point_list

            else:
                if noise_frame_key == 'pc0_all':
                    point_condition = pc0_condition.dense().permute(0,2,3,4,1) #b,x,y,z,c
                    change_frame_key = 'pc0s'
                elif noise_frame_key == 'pc1_all':
                    point_condition = pc1_condition.dense().permute(0,2,3,4,1) #b,x,y,z,c
                    change_frame_key = 'pc1s'
                
                pc_all = input_dict[change_frame_key]
                voxel_info_list_all = self.voxelizer_4(pc_all)
                # result_dict = {
                #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
                #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
                #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
                #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
                # }
                restore_point_list = []
                for b_idx, voxel_info_dict_all in enumerate(voxel_info_list_all):
                    coordinates_all = voxel_info_dict_all['voxel_coords'].repeat(5,1) #! z,y,x
                    batch_indices_all = torch.full((coordinates_all.size(0), 1), b_idx, dtype=torch.long, device=voxel_coors.device)
                    voxel_coors_b = torch.cat([batch_indices_all, coordinates_all[:, [2, 1, 0]]], dim=1) 
                    #! voxel中心
                    voxel_center_xyz = self.offset_voxel_center(None, coordinates_all)
                    cond_voxel_to_point = point_condition[voxel_coors_b[:, 0], voxel_coors_b[:, 1], voxel_coors_b[:, 2], voxel_coors_b[:, 3], :]
                    mask_para = ~(cond_voxel_to_point.any(dim =1))
                    cond_voxel_to_point = cond_voxel_to_point * ~mask_para[:, None] + mask_para[:, None] * self.zeromask # N, 64

                    #! voxel位置编码
                    pos_embed = self.pos_embed(voxel_center_xyz) # N, 32
                    time_embed = time_code[b_idx].repeat(cond_voxel_to_point.shape[0], 1) # N, 4
                    #! 全部条件
                    cond_all_in = torch.cat([cond_voxel_to_point, pos_embed, time_embed], dim=1) # N, 68
                    #! 点云offset加噪声
                    noise =  torch.randn((coordinates_all.size(0), 3), device= device).float()
                    noise = noise * (self.vx/2) #0.4
                    # out = self.var_sched.q_sample(x_start=points_offset_voxel_center, t=ts[b_idx], noise = noise)
                    xt = noise
                    out = noise
                    for i, layer in enumerate(self.layers):
                        out = layer(fea=out, cond=cond_all_in)
                        if i < len(self.layers) - 1:
                            out = self.act(out)
                    pred_out = out
                    # pred_out = self.var_sched.predict_start_from_noise(x_t = xt, t=ts[b_idx], noise = pred_noise)  # self, x_t, t, noise)
                    restore_point = torch.clamp(pred_out, min = -1.0*self.vx/2, max = 1.0*self.vx/2) + voxel_center_xyz
                    restore_point_list.append(restore_point)
                restore_point_dict[noise_frame_key] = restore_point_list
                
        # restore_point_dict['pc0_all'] = input_dict['pc0s']
        # restore_point_dict['pc1_all'] = input_dict['pc1s']
        if training_flag:
            return restore_point_dict, total_loss
        else:
            return restore_point_dict
    


class DynamicEmbedder_4D_restore(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        # self.scatter = PointPillarsScatter(in_channels=feat_channels,
        #                                    output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s_restore', 'pc1s_restore', 'pc0s']

        pc0_point_feats_lst = []

        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4

            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)

            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp_4d)  #batch_idx, x,y,z, time_idx 

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]

        all_voxel_feats_sp = torch.cat(voxel_feats_list[:2], dim=0)
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list[:2], dim=0)

        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), self.voxel_spatial_shape, int(batch_index + 1))

        output = {
            '4d_tensor': sparse_tensor_4d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_mum_voxels': pc0s_num_voxels # 第0帧包括所有batch的voxel数量
        }

        return output
    

class DynamicEmbedder_4D_pc0(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, pc) -> torch.Tensor:



        pc0_point_feats_lst = []

        voxel_info_list = self.voxelizer(pc)
        # result_dict = {
        #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
        #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
        #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
        #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
        # }
        voxel_feats_list_batch = []

        for batch_index, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)

            # if frame_key == 'pc0s':
            pc0_point_feats_lst.append(point_feats)
            voxel_feats_list_batch.append(voxel_feats)
            # voxel_coors_list_batch.append(voxel_coors_batch)

        voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接

        pc0s_3dvoxel_infos_lst = voxel_info_list
        pc0s_num_voxels = voxel_feats_sp.shape[0]

        output = {
            # '4d_tensor': sparse_tensor_4d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_mum_voxels': pc0s_num_voxels # 第0帧包括所有batch的voxel数量
        }

        return output
    

class DynamicEmbedder_4D_ori(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet_flow4D(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)
        
        self.voxel_spatial_shape = pseudo_image_dims

    def forward(self, input_dict, training_flag) -> torch.Tensor:
        voxel_feats_list = []
        voxel_coors_list = []
        batch_index = 0

        frame_keys = sorted([key for key in input_dict.keys() if key.startswith('pc_m')], reverse=True)
        frame_keys += ['pc0s', 'pc1s']

        pc0_point_feats_lst = []

        for time_index, frame_key in enumerate(frame_keys):
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)
            # result_dict = {
            #     "points": valid_batch_non_nan_points, #除去nan以及不在point_range内的点
            #     "voxel_coords": valid_batch_voxel_coords, #有效点对应的voxel坐标
            #     "point_idxes": valid_point_idxes, #每个有效点对应的voxel内的索引
            #     "point_offsets": point_offsets #每个有效点对应的voxel中心的偏移量
            # }
            voxel_feats_list_batch = []
            voxel_coors_list_batch = []

            for batch_index, voxel_info_dict in enumerate(voxel_info_list):
                points = voxel_info_dict['points']
                coordinates = voxel_info_dict['voxel_coords']
                voxel_feats, voxel_coors, point_feats = self.feature_net(points, coordinates)

                if frame_key == 'pc0s':
                    pc0_point_feats_lst.append(point_feats)
                    
                batch_indices = torch.full((voxel_coors.size(0), 1), batch_index, dtype=torch.long, device=voxel_coors.device)
                voxel_coors_batch = torch.cat([batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)

                voxel_feats_list_batch.append(voxel_feats)
                voxel_coors_list_batch.append(voxel_coors_batch)

            voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0) #N*16 在0维度进行B的拼接
            coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(dtype=torch.int32) #N*4

            time_dimension = torch.full((coors_batch_sp.shape[0], 1), time_index, dtype=torch.int32, device='cuda')
            coors_batch_sp_4d = torch.cat((coors_batch_sp, time_dimension), dim=1)

            voxel_feats_list.append(voxel_feats_sp) #不同time下的voxel特征
            voxel_coors_list.append(coors_batch_sp_4d)  #batch_idx, x,y,z, time_idx 

            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]

        all_voxel_feats_sp = torch.cat(voxel_feats_list, dim=0)
        all_coors_batch_sp_4d = torch.cat(voxel_coors_list, dim=0)

        sparse_tensor_4d = spconv.SparseConvTensor(all_voxel_feats_sp.contiguous(), all_coors_batch_sp_4d.contiguous(), self.voxel_spatial_shape, int(batch_index + 1))

        output = {
            '4d_tensor': sparse_tensor_4d,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_mum_voxels': pc0s_num_voxels # 第0帧包括所有batch的voxel数量
        }

        return output
