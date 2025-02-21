"""
# Created: 2023-11-01 17:02
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Preprocess Data, save as h5df format for faster loading
# Reference: 
#   * ZeroFlow data preprocessing work: https://github.com/kylevedder/argoverse2-sf
#   * Argoverse API source code: https://github.com/argoverse/av2-api
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
from av2.structures.sweep import Sweep
from av2.structures.cuboid import CuboidList, Cuboid
from av2.utils.io import read_feather
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3
from av2.datasets.sensor.constants import AnnotationCategories

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union, Final
from tqdm import tqdm
import numpy as np
import fire, time, h5py
from collections import defaultdict
import pickle
from zipfile import ZipFile
import pandas as pd
import torch

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index
from scripts.network.models.basic.make_voxels import DynamicVoxelizer


BOUNDING_BOX_EXPANSION: Final = 0.2
CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}


#! 全局voxel参数化定义
voxel_size = [0.2, 0.2, 0.2]
point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2]
grid_feature_size = [512, 512, 32]
voxel_size_32 = [v * 32 for v in voxel_size]
voxel_spatial_shape_32 = [int(v / 32) for v in grid_feature_size]

# **只初始化一次**
global_voxelizer = DynamicVoxelizer(voxel_size=voxel_size_32, point_cloud_range=point_cloud_range)


def create_eval_mask(data_mode: str, output_dir_: Path, mask_dir: str):
    """
    Need download the official mask file run: `s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .`
    Check more in our assets/README.md
    """
    # mask_file_path = Path(mask_dir) / f"{data_mode}-masks.zip"
    # if not mask_file_path.exists():
    #     print(f'{mask_file_path} not found, please download the mask file for official evaluation.')
    #     return
    # # extract the mask file
    # with ZipFile(mask_file_path, 'r') as zipObj:
    #     zipObj.extractall(Path(mask_dir) / f"{data_mode}-masks")
    
    data_index = []
    # list scene ids
    scene_ids = os.listdir(Path(mask_dir) / f"{data_mode}-masks")
    for scene_id in tqdm(scene_ids, desc=f'Create {data_mode} eval mask', ncols=100):
        timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(Path(mask_dir) / f"{data_mode}-masks" / scene_id)
                        if file.endswith('.feather')])
        with h5py.File(output_dir_ / f'{scene_id}.h5', 'r+') as f:
            for ts in timestamps:
                key = str(ts)
                if key not in f.keys():
                    print(f'{scene_id}/{key} not found')
                    continue
                group = f[key]
                mask = pd.read_feather(Path(mask_dir) / f"{data_mode}-masks" / scene_id / f"{key}.feather").to_numpy().astype(bool)
                group.create_dataset('eval_mask', data=mask)
                data_index.append([scene_id, key])

    with open(output_dir_/'index_eval.pkl', 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create reading index Successfully")

def read_pose_pc_ground(data_dir: Path, log_id: str, timestamp: int, avm: ArgoverseStaticMap):
    log_poses_df = read_feather(data_dir / log_id / "city_SE3_egovehicle.feather")
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin([timestamp])]
    pose = convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == timestamp])
    # pc = Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{timestamp}.feather").xyz
    # transform to city coordinate since sweeps[0].xyz is in ego coordinate to get ground mask
    # is_ground = avm.get_ground_points_boolean(pose.transform_point_cloud(pc))
    return pose #pc, pose, is_ground

def compute_sceneflow(data_dir: Path, log_id: str, timestamps: Tuple[int, int], avm: ArgoverseStaticMap, stage: str, ts0: int) -> Dict[str, Union[np.ndarray, SE3]]:
    """Compute sceneflow between the sweeps at the given timestamps.
        Args:
          data_dir: Argoverse 2.0 directory, e.g. /home/kin/data/av2/sensor/train
          log_id: unique id.
          timestamps: the timestamps of the lidar sweeps to compute flow between
        Returns:
          Dictionary with fields:
            pcl_0: Nx3 array containing the points at time 0
            pcl_1: Mx3 array containing the points at time 1
            flow_0_1: Nx3 array containing flow from timestamp 0 to 1
            flow_1_0: Mx3 array containing flow from timestamp 1 to 0
            valid_0: Nx1 array indicating if the returned flow from 0 to 1 is valid (1 for valid, 0 otherwise)
            valid_1: Mx1 array indicating if the returned flow from 1 to 0 is valid (1 for valid, 0 otherwise)
            classes_0: Nx1 array containing the class ids for each point in sweep 0
            classes_1: Nx1 array containing the class ids for each point in sweep 0
            pose_0: SE3 pose at time 0
            pose_1: SE3 pose at time 1
            ego_motion: SE3 motion from sweep 0 to sweep 1
    """
    def accumulate_cuboids(sweeps, cuboids, poses,  ground_masks, stage): # include 5 frame
        if stage == 'start':
            target_idx = 0
        elif stage == 'end':
            target_idx = -2
        elif stage == 'mid':
            target_idx = len(sweeps) // 2
        elif stage == 'last':
            target_idx = -1
    
        
        classes_pts_list = []
        transformed_pts_list = []
        valid_pts_list = []
        ground_masks_list = []
        
        for time_idx in range(len(sweeps)):
            #！当为目标帧时，后续直接使用原始点云
            if time_idx == target_idx: 
                continue
            #! 背景静态点云
            ego1_SE3_ego0 = poses[target_idx].inverse().compose(poses[time_idx])
            # Convert to float32s
            ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
            ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
            
            transformed_point = ego1_SE3_ego0.transform_point_cloud(sweeps[time_idx].xyz) #! 静态初始化
            # Convert to float32s
            transformed_point = transformed_point.astype(np.float32)
            
            valid = np.ones(len(sweeps[time_idx].xyz), dtype=np.bool_)
            classes = np.zeros(len(sweeps[time_idx].xyz), dtype=np.uint8)
            ground_masks_idx =  ground_masks[time_idx]
            for id in cuboids[time_idx]:
                c0 = cuboids[time_idx][id]
                c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
                c0.width_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(sweeps[time_idx].xyz) #! obj_mask的都不属于背景
                classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
                if id in cuboids[target_idx]:
                    # classes_pts = np.full(obj_pts.shape[0], CATEGORY_TO_INDEX[str(c0.category)], dtype=np.uint8)

                    # if time_idx == target_idx:
                    #     transformed_pts = obj_pts.astype(np.float32) #!当为目标帧时，直接使用原始点云
                    # else:
                    c1 = cuboids[target_idx][id]
                    c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                    transformed_pts = c1_SE3_c0.transform_point_cloud(obj_pts).astype(np.float32)
                     
                    # classes_pts_list.append(classes_pts)
                    transformed_point[obj_mask] = transformed_pts #! 添加box中的点
                else:
                    valid[obj_mask] = 0 #! 背景点被视为无效点
             
            transformed_pts_list.append(transformed_point) #! 添加变换后的点
            valid_pts_list.append(valid) #! 点云有效性
            ground_masks_list.append(ground_masks_idx) #! 添加地面点
            classes_pts_list.append(classes) #! 添加类别标签
        classes_pts_all =np.concatenate(classes_pts_list, axis=0)   
        transformed_pts_others =np.concatenate(transformed_pts_list, axis=0) 
        valid_pts_others = np.concatenate(valid_pts_list, axis=0)
        ground_masks_others = np.concatenate(ground_masks_list, axis=0)
    

        return transformed_pts_others, valid_pts_others, ground_masks_others, target_idx, classes_pts_all


    def compute_flow(sweeps, cuboids, poses, ground_masks, target_idx, transformed_pts_others, valid_pts_others, ground_masks_others, classes_others):

        ego1_SE3_ego0 = poses[target_idx+1].inverse().compose(poses[target_idx])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        flows_list = []
        classes_list = [] 
        valid_flow_list = []
        valid_point_list = []
        ground_point_list = []
        target_mask_list = []
        point_all_list = []
        class_valid_list = []
        #! 其他帧点云
        for key_name in ['others', 'target_pc0']:
            if key_name == 'others':
                new_pc0 = transformed_pts_others
                target_mask = np.zeros(len(new_pc0), dtype=np.bool_)
                valid_point = valid_pts_others
                ground_point = ground_masks_others
            elif key_name == 'target_pc0':
                new_pc0 = sweeps[target_idx].xyz
                target_mask = np.ones(len(new_pc0), dtype=np.bool_)
                valid_point = np.ones(len(new_pc0), dtype=np.bool_) #! 目标帧全部点云有效性
                ground_point = ground_masks[target_idx]

            flow = ego1_SE3_ego0.transform_point_cloud(new_pc0) -  new_pc0
            # Convert to float32s
            flow = flow.astype(np.float32)
            
            valid = np.ones(len(new_pc0), dtype=np.bool_)
            # classes = -np.ones(len(sweeps[0].xyz), dtype=np.int8)
            classes = np.zeros(len(new_pc0), dtype=np.uint8)

            
            for id in cuboids[target_idx]:
                c0 = cuboids[target_idx][id]
                c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
                c0.width_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(new_pc0)
                classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
            
                if id in cuboids[target_idx + 1]:
                    c1 = cuboids[target_idx + 1][id]
                    c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                    obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                    flow[obj_mask] = obj_flow.astype(np.float32)
                else:
                    valid[obj_mask] = 0

            if key_name == 'others':
                class_valid = (classes_others == classes)
            else:
                class_valid = np.ones(len(classes), dtype=np.bool_)

            flows_list.append(flow)
            classes_list.append(classes)
            valid_flow_list.append(valid)
            valid_point_list.append(valid_point)
            ground_point_list.append(ground_point)
            target_mask_list.append(target_mask)
            point_all_list.append(new_pc0)
            class_valid_list.append(class_valid)


        flows_cat = np.concatenate(flows_list, axis=0)
        classes_cat = np.concatenate(classes_list, axis=0)
        valid_flow_cat = np.concatenate(valid_flow_list, axis=0)
        valid_point_cat = np.concatenate(valid_point_list, axis=0)
        ground_point_cat = np.concatenate(ground_point_list, axis=0)
        target_mask_cat = np.concatenate(target_mask_list, axis=0)
        point_all_cat = np.concatenate(point_all_list, axis=0)
        class_valid_cat = np.concatenate(class_valid_list, axis=0)
        
        return point_all_cat, flows_cat, classes_cat, valid_flow_cat, valid_point_cat, ground_point_cat, target_mask_cat, ego1_SE3_ego0, class_valid_cat

    sweeps = [Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{ts}.feather") for ts in timestamps]


    # ================== Load annotations ==================
    annotations_feather_path = data_dir / log_id / "annotations.feather"
    
    if not annotations_feather_path.exists():
        # print(f'{annotations_feather_path} not found')
        timestamp_cuboid_index = {}
    else:
        # Load annotations from disk.
        # NOTE: This file contains annotations for the ENTIRE sequence.
        # The sweep annotations are selected below.
        cuboid_list = CuboidList.from_feather(annotations_feather_path)

        raw_data = read_feather(annotations_feather_path)
        ids = raw_data.track_uuid.to_numpy()
        timestamp_cuboid_index = defaultdict(dict)
        for id, cuboid in zip(ids, cuboid_list.cuboids):
            timestamp_cuboid_index[cuboid.timestamp_ns][id] = cuboid
    # ================== Load annotations ==================

    cuboids = [timestamp_cuboid_index.get(ts, {}) for ts in timestamps]

    log_poses_df = read_feather(data_dir / log_id / "city_SE3_egovehicle.feather")
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin(timestamps)]
    poses = [convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == ts]) for ts in timestamps]
    #! ground_mask
    ground_masks = [avm.get_ground_points_boolean(poses[i].transform_point_cloud(sweeps[i].xyz)) for i in range(len(sweeps))]


    transformed_pts_others, valid_pts_others, ground_masks_others, target_idx, classes_others = accumulate_cuboids(sweeps, cuboids, poses, ground_masks, stage) #! 累积点云
    assert ts0 == timestamps[target_idx]
    if stage == 'last':
        assert target_idx == -1
        point_last = sweeps[-1].xyz
        valid_point_last = np.ones(len(point_last), dtype=np.bool_)
        ground_point_last = ground_masks[-1]
        point_cat = np.concatenate([transformed_pts_others, point_last], axis=0)
        valid_point_cat = np.concatenate([valid_pts_others, valid_point_last], axis=0)
        ground_point_cat = np.concatenate([ground_masks_others, ground_point_last], axis=0)
        non_target_mask = np.zeros(len(transformed_pts_others), dtype=np.bool_)
        point_target_mask = np.ones(len(point_last), dtype=np.bool_)
        target_mask_1 = np.concatenate([non_target_mask, point_target_mask], axis=0)

        return {'acc_pc0': point_cat, 'valid_point': valid_point_cat, 'ground_point': ground_point_cat, 'target_mask': target_mask_1}
    point_cat, flows_cat, classes_cat, valid_flow_cat, valid_point_cat, ground_point_cat, target_mask_cat, ego1_SE3_ego0, class_valid_cat = \
                compute_flow(sweeps, cuboids,  poses, ground_masks, target_idx, transformed_pts_others, valid_pts_others, 
                             ground_masks_others, classes_others)

    # return {'pcl_0': sweeps[0].xyz, 'pcl_1' :sweeps[1].xyz, 'flow_0_1': flow_0_1,
    #         'valid_0': valid_0, 'classes_0': classes_0, 
    #         'pose_0': poses[0], 'pose_1': poses[1],
    #         'ego_motion': ego_motion, 'new_pc0': transformed_pts_all}
    return {'acc_pc0': point_cat, 'flow_0_1': flows_cat, 'classes_0': classes_cat,
            'valid_flow_0': valid_flow_cat, 'valid_point': valid_point_cat, 'ground_point': ground_point_cat, 
            'target_mask': target_mask_cat, 'ego_motion': ego1_SE3_ego0, 'class_valid': class_valid_cat}

def fast_knn_gpu_9_idx(sparse_points, sparse_voxels, dense_points, dense_voxels, K):
    """
    以 Voxel 为基准，使用 PyTorch GPU 进行高效 KNN 搜索，
    每个稀疏点在水平方向 (xy) 附近体素内找到最近的 K 个稠密点，返回索引。

    Returns:
        torch.Tensor: (N, K) 形状的 Tensor,存放在 GPU 上，表示最近邻的索引。
    """
    device = sparse_points.device
    N, M = sparse_points.shape[0], dense_points.shape[0]

    # 1. 计算 Voxel 索引 #! voxel coor 是 zyx
    voxel_spatial_shape = voxel_spatial_shape_32  
    sparse_voxel_keys = (sparse_voxels[:, 0] * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
                        sparse_voxels[:, 1] * voxel_spatial_shape[0] + sparse_voxels[:, 2])
    dense_voxel_keys = (dense_voxels[:, 0] * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
                        dense_voxels[:, 1] * voxel_spatial_shape[0] + dense_voxels[:, 2])

    # 2. 获取唯一的 sparse_voxel_keys 和其索引
    unique_sparse_voxels, inverse_indices = torch.unique(sparse_voxels, return_inverse=True, dim=0)
    # 返回 原始 sparse_voxels 里的每个体素对应 unique_sparse_voxels 的索引
    # 形状是 (N,)，表示 sparse_voxels[i] 对应 unique_sparse_voxels[inverse_indices[i]]

    # 3. 计算所有 (dx, dy) 偏移的 9 邻域
    neighbor_shifts = torch.tensor([
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ], device=device, dtype=torch.int32)  # (9, 2)

    # 4. **一次性计算所有 unique sparse 的 9 邻域**
    new_voxels_x = unique_sparse_voxels[:, 2].unsqueeze(1) + neighbor_shifts[:, 0]  # (U, 9)  x
    new_voxels_y = unique_sparse_voxels[:, 1].unsqueeze(1) + neighbor_shifts[:, 1]  # (U, 9)  y

    # 5. **计算合法 mask，确保 x 和 y 在范围内**
    valid_mask = (new_voxels_x >= 0) & (new_voxels_x < voxel_spatial_shape[0]) & \
                (new_voxels_y >= 0) & (new_voxels_y < voxel_spatial_shape[1])

    # 6. **计算新的 voxel key**
    expanded_sparse_voxel_keys = (
        unique_sparse_voxels[:, 0].unsqueeze(1) * voxel_spatial_shape[0] * voxel_spatial_shape[1] +
        new_voxels_y * voxel_spatial_shape[0] + new_voxels_x
    )  # (U, 9)

    # 7. **仅保留合法的 voxel keys**
    expanded_sparse_voxel_keys = expanded_sparse_voxel_keys[valid_mask]

    # 8. **计算每个 unique sparse voxel 的实际邻域数**
    num_valid_neighbors = valid_mask.sum(dim=1)  # (U,)
    neighbor_offsets = torch.cat([torch.tensor([0], device=device), num_valid_neighbors.cumsum(dim=0)])

    # 9. 对 Dense Voxel 进行排序
    dense_sorted_indices = torch.argsort(dense_voxel_keys)
    sorted_dense_keys = dense_voxel_keys[dense_sorted_indices]

    # 使用二分查找 (torch.searchsorted) 在 sorted_dense_keys 中找到 expanded_sparse_voxel_keys 的起始和结束索引
    voxel_start_idx = torch.searchsorted(sorted_dense_keys, expanded_sparse_voxel_keys)
    voxel_end_idx = torch.searchsorted(sorted_dense_keys, expanded_sparse_voxel_keys, side="right")

    # 11. 预分配 (N, K) 形状的索引 Tensor
    results = torch.full((N, K), -1, device=device, dtype=torch.long)  # -1 作为无效索引填充

    # 12. **并行 KNN 计算**
    for i in range(len(unique_sparse_voxels)):
        sparse_indices = torch.where(inverse_indices == i)[0]  # 当前 voxel 内所有 sparse 点索引
        sparse_pts = sparse_points[sparse_indices]  # (Nsparse, 3)

        # 获取实际邻域范围
        start_idx_list = voxel_start_idx[neighbor_offsets[i]:neighbor_offsets[i+1]]
        end_idx_list = voxel_end_idx[neighbor_offsets[i]:neighbor_offsets[i+1]]

        dense_indices = torch.cat([dense_sorted_indices[start:end] for start, end in zip(start_idx_list, end_idx_list)])

        if len(dense_indices) > 0:
            dense_pts = dense_points[dense_indices]  # (Ndense, 3)

            # **计算 KNN**
            distances = torch.cdist(sparse_pts, dense_pts)  # (Nsparse, Ndense)
            knn_indices = distances.topk(k=min(K, len(dense_pts)), dim=1, largest=False).indices  # 取最近 K 个

            # **填充 K 个邻居**
            num_neighbors = knn_indices.shape[1]
            if num_neighbors < K:
                repeat_indices = torch.randint(0, num_neighbors, (K - num_neighbors,), device=device)
                knn_indices = torch.cat([knn_indices, knn_indices[:, repeat_indices]], dim=1)

            # **存储索引**
            results[sparse_indices] = dense_indices[knn_indices]  # 存储的是 dense 点的全局索引

    return results 

def sparse_to_dense(data_dict, neighbor_num, flow_flag):

    pc0 = torch.tensor(data_dict['acc_pc0'][:], dtype=torch.float32) #累积后的点云
    gm0 = torch.tensor(data_dict['ground_point'][:], dtype=torch.bool) #地面点mask
    pc0_valid = torch.tensor(data_dict['valid_point'][:], dtype=torch.bool) #有效点云


    #! 只用修改单帧
    pc0_valid_mask = (~gm0 & pc0_valid) #有效的pc0_all mask
    target_mask_pc0 = torch.tensor(data_dict['target_mask'][:], dtype=torch.bool) 

    pc0_origin = pc0[target_mask_pc0]
    pc0_gm0_origin = gm0[target_mask_pc0]
    pc0_all = pc0[pc0_valid_mask][None].cuda()
    pc0_one = pc0_origin[~pc0_gm0_origin][None].cuda()

    voxel_info_dict_all = global_voxelizer(pc0_all)
    voxel_info_dict_all_sparse = global_voxelizer(pc0_one)
    #
    points_all = voxel_info_dict_all[0]['points']
    coordinates_all = voxel_info_dict_all[0]['voxel_coords'] #! z,y,x
    point_idxes_all = voxel_info_dict_all[0]['point_idxes']
    #
    points_sparse = voxel_info_dict_all_sparse[0]['points']
    coordinates_sparse = voxel_info_dict_all_sparse[0]['voxel_coords'] #! z,y,x
    point_idxes_sparse = voxel_info_dict_all_sparse[0]['point_idxes']


    neighbor_idx = fast_knn_gpu_9_idx(points_sparse, coordinates_sparse, points_all, coordinates_all, neighbor_num) # N,K,3
    # 稀疏点的邻居信息
    neighbor_point = points_all[neighbor_idx]
    indices_A_to_C_all = point_idxes_all[neighbor_idx]
    assert torch.allclose(points_all[neighbor_idx], pc0_all[0][indices_A_to_C_all])
    indices_A_to_B = (~pc0_gm0_origin).nonzero(as_tuple=True)[0]
    indices_A_to_C = indices_A_to_B[point_idxes_sparse.cpu()]
    assert torch.allclose(points_sparse.cpu(), pc0_origin[indices_A_to_C])
    assert neighbor_point.shape[0] == indices_A_to_C.shape[0]
    if flow_flag:
        #! flow class 的有效mask
        flow_another_mask = torch.tensor(data_dict['class_valid'][:], dtype=torch.bool)
        flow_is_valid = torch.tensor(data_dict['valid_flow_0'][:], dtype=torch.bool)
        flow_final_mask = flow_is_valid & flow_another_mask #有效的flow mask
        point_idxes_all = point_idxes_all.cpu()
        flow_masks_all = flow_final_mask[point_idxes_all]
        flow_value_all = data_dict['flow_0_1'][point_idxes_all]
        flow_category_all = data_dict['classes_0'][point_idxes_all]
        neighbor_idx = neighbor_idx.cpu()
        neighbor_flow_valid = flow_masks_all[neighbor_idx]
        neighbor_flow = flow_value_all[neighbor_idx]
        neighbor_flow_category = flow_category_all[neighbor_idx]
        return {'neighbor_point': neighbor_point.cpu().numpy(), 'indices_A_to_C': indices_A_to_C.numpy(),
                'neighbor_flow_valid': neighbor_flow_valid.numpy(), 'neighbor_flow': neighbor_flow,
                'neighbor_flow_category': neighbor_flow_category}
    else:
        return {'neighbor_point': neighbor_point.cpu().numpy(), 'indices_A_to_C': indices_A_to_C.numpy()}



def all_to_one(data_dict, flow_flag):
    #! 单帧
    target_mask_pc0 = data_dict['target_mask']
    target_point = data_dict['acc_pc0'][target_mask_pc0]
    target_point_ground = data_dict['ground_point'][target_mask_pc0]
    if flow_flag:
        target_flow = data_dict['flow_0_1'][target_mask_pc0]
        target_flow_valid = data_dict['valid_flow_0'][target_mask_pc0]
        target_flow_category = data_dict['classes_0'][target_mask_pc0]

        return {'target_point': target_point, 
            'target_point_ground': target_point_ground, 'target_flow': target_flow,
            'target_flow_valid': target_flow_valid, 'target_flow_category': target_flow_category}
    else:
        return {'target_point': target_point, 
            'target_point_ground': target_point_ground}




def process_log(data_dir: Path, log_id: str, output_dir: Path, multi_frame: int, neighbor_num: int, n: Optional[int] = None) :

    def create_group_data(group, target_point, pose, target_ground_mask, neighbor_point, neighbor_a_to_c, neighbor_flow_vaild = None,
                          neighbor_flow = None, neighbor_flow_category = None,
                          target_flow = None, target_flow_valid = None, target_flow_category = None,
                          ego_motion=None):
        # if pc is not None:
        group.create_dataset('target_point', data=target_point.astype(np.float32))
        group.create_dataset('target_ground_mask', data=target_ground_mask.astype(bool))
        group.create_dataset('neighbor_point', data=neighbor_point.astype(np.float32))
        group.create_dataset('neighbor_a_to_c', data=neighbor_a_to_c.astype(np.int64))
        group.create_dataset('pose', data=pose.astype(np.float32))

        if neighbor_flow_vaild is not None:
            # ground truth flow information
            group.create_dataset('neighbor_flow', data=neighbor_flow.astype(np.float32))
            group.create_dataset('neighbor_flow_vaild', data=neighbor_flow_vaild.astype(bool))
            group.create_dataset('neighbor_flow_category', data=neighbor_flow_category.astype(np.uint8))
            group.create_dataset('target_flow', data=target_flow.astype(np.float32))
            group.create_dataset('target_flow_valid', data=target_flow_valid.astype(bool))
            group.create_dataset('target_flow_category', data=target_flow_category.astype(np.uint8))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))

    log_map_dirpath = data_dir / log_id / "map"
    if(len(os.listdir(log_map_dirpath))<3):
        print(f'{log_map_dirpath} needed by 3 to find the ground layer, check if you are using the correct *sensor* dataset')
        print("If you are using *lidar* dataset, Please run the following command to generate the map files:")
        print(f"python run_steps/0_additional_lidar_map.py --argo_dir {data_dir}")
        return
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(data_dir / log_id / "sensors/lidar")
                        if file.endswith('.feather')])

    # if n is not None:
    #     iter_bar = tqdm(zip(timestamps, timestamps[1:]), leave=False,
    #                      total=len(timestamps) - 1, position=n,
    #                      desc=f'Log {log_id}')
    # else:
    #     iter_bar = zip(timestamps, timestamps[1:])

    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, ts0 in enumerate(timestamps):
            # if str(ts0) in f: #! for debug
            #     continue
            # else:
            group = f.create_group(str(ts0))
            pose0 = read_pose_pc_ground(data_dir, log_id, ts0, avm)
            # multi_frame = 19 #! 多帧
            # print('multi_frame:', multi_frame)
            mid_frame = multi_frame // 2
            if cnt == len(timestamps) - 1:
                stage = 'last'
                scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt-multi_frame+1:], avm, stage, ts0)
                neighbor_info = sparse_to_dense(scene_flow, neighbor_num, False)
                target = all_to_one(scene_flow, False)
                create_group_data(group, target['target_point'], pose0.transform_matrix.astype(np.float32), #! change to new_pc0
                                  target['target_point_ground'].astype(np.bool_),
                                  neighbor_info['neighbor_point'], neighbor_info['indices_A_to_C']
                                  )
            else:
                if cnt < mid_frame:
                    stage = 'start'
                    scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt:cnt + multi_frame], avm, stage, ts0)
                    neighbor_info = sparse_to_dense(scene_flow, neighbor_num, True)
                    target = all_to_one(scene_flow, True)
                elif cnt >= len(timestamps) - mid_frame:
                    stage = 'end'
                    scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt-multi_frame+2:cnt+2], avm, stage, ts0)
                    neighbor_info = sparse_to_dense(scene_flow, neighbor_num, True)
                    target = all_to_one(scene_flow, True)
                else:
                    stage ='mid'
                    scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt-mid_frame:cnt+mid_frame+1], avm, stage, ts0)
                    neighbor_info = sparse_to_dense(scene_flow, neighbor_num, True)
                    target = all_to_one(scene_flow, True)
                assert scene_flow['acc_pc0'].shape[0] == scene_flow['class_valid'].shape[0]
                create_group_data(group, target['target_point'], pose0.transform_matrix.astype(np.float32), #! change to new_pc0
                                  target['target_point_ground'].astype(np.bool_),
                                  neighbor_info['neighbor_point'], neighbor_info['indices_A_to_C'],
                                  neighbor_info['neighbor_flow_valid'], neighbor_info['neighbor_flow'],
                                  neighbor_info['neighbor_flow_category'],
                                  target['target_flow'], target['target_flow_valid'], target['target_flow_category'],
                                  scene_flow['ego_motion'].transform_matrix.astype(np.float32),
                                  )

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_dir: Path, output_dir: Path, nproc: int, multi_frame: int, neighbor_num: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    
    if not data_dir.exists():
        print(f'{data_dir} not found')
        return
    
    # NOTE(Qingwen): if you don't want to all data_dir, then change here: logs = logs[:10] only 10 scene.
    logs = os.listdir(data_dir)
    args = sorted([(data_dir, log, output_dir,  multi_frame, neighbor_num) for log in logs])
    print(f'Using {nproc} processes to process data: {data_dir} to .h5 format. (#scenes: {len(args)})')
    # #! for debug
    # for x in tqdm(args):
    #     proc(x, ignore_current_process=True)
    #     break
    if nproc <= 1:
        for x in tqdm(args, ncols=120):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(logs), ncols=120))

def main(
    argo_dir: str = "/data0/dataset/av2",
    output_dir: str ="/data1/dataset/av2/debug",
    av2_type: str = "sensor",
    data_mode: str = "val",
    mask_dir: str = "/data0/dataset/av2/eval_mask",
    multi_frame: int = 5,
    neighbor_num: int = 16,
    nproc: int = (multiprocessing.cpu_count() - 1)
):
    data_root_ = Path(argo_dir) / av2_type/ data_mode
    output_dir_ = Path(output_dir) / av2_type / data_mode
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(data_root_, output_dir_, nproc, multi_frame, neighbor_num)
    create_reading_index(output_dir_)
    if data_mode == "val" or data_mode == "test":
        create_eval_mask(data_mode, output_dir_, mask_dir)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")