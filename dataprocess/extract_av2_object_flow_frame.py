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

# 存储信息
dataset_root/
├── scene_001/                  # 场景1
│   ├── box_id_0001/            # 实例1(按box_id划分)
│   │   ├── first_seen_frame    # 该box出现的第一帧
|   |   ├── last_seen_frame     # 该box出现的最后一帧
|   |   ├── frames          # 该box的总帧数列表
│   │   ├── frames_0001/        # 实例的点云序列（按时间帧存储）
│   │   │   ├── point  # 时间帧1的点云
│   │   │   ├── pose   # box在该时间帧下的pose
│   │   │   ├── ego_pose   # 该时间帧下的ego_pose
│   │   │   ├── vertices   # box的顶点坐标
│   │   │   ├── lwh_center     # box的长宽高、中心坐标
│   │   │   ├── category     # 类别
│   │   │   ├── dynamic_status    # 动静状态
│   │   │   └── valid_status    # 动静有效状态  相邻帧消失
│   │   ├── frames_0002/        # 实例的点云序列（按时间帧存储）
│   │   │   ├── point  # 时间帧2的点云
│   │   │   ├── pose   # box在该时间帧下的pose
│   │   │   ├── ...   # 其他信息
│   │   ├── acc_frames/
|   |   |   ├── acc_point  # 该box的时序累积点云acc_point
|   |   |   ├── category   # 类别
|   |   |   ├── dynamic_status    # 动静状态
|   |   |   └── valid_status    # 动静有效状态  相邻帧消失
│   ├── box_id_0002/            # 实例2
│   ├── box_id_0003/            # 实例3
│   └── ...
├── scene_002/                  # 场景2
├── ...
└── global_metadata.json        # 全局元数据






"""

#被修改成查看bbox以及距离由近到远的可视化
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
import math

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_mixdata_frame
import bisect

BOUNDING_BOX_EXPANSION: Final = 0.2
CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}

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
    pc = Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{timestamp}.feather").xyz
    # transform to city coordinate since sweeps[0].xyz is in ego coordinate to get ground mask
    is_ground = avm.get_ground_points_boolean(pose.transform_point_cloud(pc))
    return pc, pose, is_ground


def angle_with_x_axis_2d(point):
    """
    计算二维平面中点与 x 轴的夹角（单位：度，范围 0°~360°）
    
    参数：
        point (tuple/list): 二维坐标点，格式 (x, y)
        
    返回：
        float: 与 x 轴的夹角，保留两位小数
        
    异常：
        ValueError: 如果输入点是原点 (0,0)
    """
    x, y = point
    if x == 0 and y == 0:
        raise ValueError("输入点不能是原点 (0,0)")
    
    # 计算弧度（范围 -π 到 π）
    radians = math.atan2(y, x)
    
    # 转换为角度（范围 -180° 到 180°）
    degrees = math.degrees(radians)
    
    # 调整到 0°~360° 范围
    if degrees < 0:
        degrees += 360.0
    
    # 划分角度区间
    interval_id = int(degrees // 45)
    # 处理 360° 边界（归入第0区间）
    if interval_id >= 8:
        interval_id = 0

    return round(degrees, 2), interval_id


def compute_sceneflow(data_dir: Path, log_id: str, timestamps: Tuple[int, int], avm: ArgoverseStaticMap) -> Dict[str, Union[np.ndarray, SE3]]:
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
    def compute_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        
        flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0]) -  sweeps[0]
        # Convert to float32s
        flow = flow.astype(np.float32)
        ego_flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0]) -  sweeps[0]
        ego_flow = ego_flow.astype(np.float32)
        
        valid = np.ones(len(sweeps[0]), dtype=np.bool_)
        # classes = -np.ones(len(sweeps[0].xyz), dtype=np.int8)
        classes = np.zeros(len(sweeps[0]), dtype=np.uint8)
        classes_box = -np.ones(len(sweeps[0]), dtype=np.int32)
        bbox_id_to_vertices = list()
        bbox_id_mapping = list() #记录box_id
        bbox_id_to_category = list()
        bbox_id_to_point = list() #记录box_id对应的点
        bbox_id_to_pose = list() #记录box_id对应的pose
        bbox_id_to_ego_pose = list() 
        bbox_id_to_dynamic_status = list() 
        bbox_id_to_valid_status = list() 
        bbox_id_to_lwh_center = list() 
        bbox_id_to_point_num = list() 
        bbox_id_to_view_id = list() 
        bbox_id_to_distance = list() 
        for box_idx, id in enumerate(cuboids[0]):
            c0 = cuboids[0][id]
            c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
            c0.width_m += BOUNDING_BOX_EXPANSION
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0])
            point_num = obj_mask.sum()
            if obj_mask.sum() == 0:  #点数太少跳过
                continue  
            c0_lwh_center = np.concatenate((c0.dims_lwh_m, c0.dst_SE3_object.translation))
            classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
            classes_box[obj_mask] = box_idx
            _, view_id = angle_with_x_axis_2d(c0.dst_SE3_object.translation[:2])
            distance = np.linalg.norm(c0.dst_SE3_object.translation)
            bbox_id_to_vertices.append(c0.vertices_m)  
            bbox_id_mapping.append(id)
            bbox_id_to_category.append(CATEGORY_TO_INDEX[str(c0.category)])
            bbox_id_to_point.append(obj_pts)
            bbox_id_to_pose.append(c0.dst_SE3_object.transform_matrix.astype(np.float32)) #从该box坐标系转换到全局坐标系
            bbox_id_to_ego_pose.append(poses[0])
            bbox_id_to_lwh_center.append(c0_lwh_center)
            bbox_id_to_point_num.append(point_num)
            bbox_id_to_view_id.append(view_id)
            bbox_id_to_distance.append(distance)
        
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                flow[obj_mask] = obj_flow.astype(np.float32)
                ego_flow_obj = ego_flow[obj_mask]
                speeds = np.linalg.norm(obj_flow - ego_flow_obj, axis=-1).mean()
                bbox_id_to_valid_status.append(True)
                if speeds > 0.04:
                    bbox_id_to_dynamic_status.append(True)
                else:
                    bbox_id_to_dynamic_status.append(False)
            else:
                valid[obj_mask] = 0
                bbox_id_to_dynamic_status.append(False)
                bbox_id_to_valid_status.append(False)

        # bbox_id_to_vertices = np.array(bbox_id_to_vertices)
        # bbox_id_mapping = np.array(bbox_id_mapping, dtype='S')
        # bbox_id_to_category = np.array(bbox_id_to_category, dtype='S')
        # return flow, classes, valid, ego1_SE3_ego0, classes_box, bbox_id_to_vertices, bbox_id_mapping, bbox_id_to_category
        return bbox_id_to_vertices, bbox_id_mapping, bbox_id_to_category, bbox_id_to_point, \
                bbox_id_to_pose, bbox_id_to_ego_pose, bbox_id_to_dynamic_status,\
                bbox_id_to_valid_status, bbox_id_to_lwh_center, bbox_id_to_point_num, bbox_id_to_view_id, bbox_id_to_distance

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
    
    #! 去除地面点
    sweeps_noground = [
        sweep_i.xyz[~avm.get_ground_points_boolean(pose_i.transform_point_cloud(sweep_i.xyz))]
        for pose_i, sweep_i in zip(poses, sweeps)
    ]


    (bbox_id_to_vertices, bbox_id_mapping, bbox_id_to_category, bbox_id_to_point, 
    bbox_id_to_pose, bbox_id_to_ego_pose, bbox_id_to_dynamic_status, bbox_id_to_valid_status, 
    bbox_id_to_lwh_center, bbox_id_to_point_num, bbox_id_to_view_id, bbox_id_to_distance) = compute_flow(sweeps_noground, cuboids, poses)

    return {'bbox_id_to_vertices': bbox_id_to_vertices, 'bbox_id_mapping': bbox_id_mapping, 'bbox_id_to_category': bbox_id_to_category, 
            'bbox_id_to_point': bbox_id_to_point, 'bbox_id_to_pose': bbox_id_to_pose, 'bbox_id_to_ego_pose': bbox_id_to_ego_pose, 
            'bbox_id_to_dynamic_status': bbox_id_to_dynamic_status, 'bbox_id_to_valid_status': bbox_id_to_valid_status,
            'bbox_id_to_lwh_center': bbox_id_to_lwh_center, 'bbox_id_to_point_num': bbox_id_to_point_num, 'bbox_id_to_view_id': bbox_id_to_view_id,
            'bbox_id_to_distance': bbox_id_to_distance}

def load_annotations(data_dir: Path, log_id: str):
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
    
    return timestamp_cuboid_index




def rank_based_dynamic_sort(boxes, alpha=0.5):
    """
    基于排名归一化的动态权重排序算法
    
    参数：
        boxes (list[Box]): 所有待排序的 Box 对象
        total (int): 需要筛选的总数
        alpha (float): 多样性权重（0~1，0=仅质量，1=仅多样性）
        
    返回：
        list[Box]: 排序后的列表
    """

    # 1. 初始质量排序：点数降序，距离升序
    sorted_by_quality = sorted(
        boxes,
        key=lambda x: (-x[0], x[1])
    )
    
    selected = []
    remaining_boxes = sorted_by_quality.copy()
    view_counts = {}  # 记录各视角已选数量


    while remaining_boxes:
        # 1. 计算质量排名（点数降序 > 距离升序）
        # 生成质量排序键：点数降序为负，距离升序不变

        quality_ranks = np.arange(len(remaining_boxes))
        quality_scores = 1.0 - (quality_ranks / (len(remaining_boxes) - 1e-9))  # 避免除以零
        
        # 2. 计算多样性排名（当前视角已选次数越少，得分越高）
        diversity_keys = [view_counts.get(box[2], 0) for box in remaining_boxes]
        # 获取多样性排名（升序排列：已选次数少的在前）
        diversity_sorted_indices = np.argsort(diversity_keys)
        diversity_ranks = np.argsort(diversity_sorted_indices)
        diversity_scores = 1.0 - (diversity_ranks / (len(remaining_boxes) - 1e-9))
        
        # 3. 计算综合得分
        total_scores = (1 - alpha) * quality_scores + alpha * diversity_scores
        
        # 4. 选择得分最高的框
        best_idx = np.argmax(total_scores)
        chosen_box = remaining_boxes.pop(best_idx)
        selected.append(chosen_box)
        view_counts[chosen_box[2]] = view_counts.get(chosen_box[2], 0) + 1
    
    return selected


frame_dtype = np.dtype([
    ('point_num', np.int32),
    ('distance', np.float32),
    ('view_id', np.int8),
    ('timestamp', 'S32')  # 固定长度的字节字符串（ASCII）
])


def list_to_np(frame_list):
    data_list = []
    for pn, dist, vid, ts in frame_list:
        # 编码为ASCII字节，截断或填充到32字节
        ts_bytes = np.string_(ts)
        data_list.append((pn, dist, vid, ts_bytes))
    data_array = np.array(data_list, dtype=frame_dtype)
    return data_array



def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None) :


    def create_group_data(f, ts0, bbox_id_to_vertices, bbox_id_mapping, bbox_id_to_category, bbox_id_to_point, bbox_id_to_pose, 
                          bbox_id_to_ego_pose, bbox_id_to_dynamic_status, bbox_id_to_valid_status, bbox_id_to_lwh_center, 
                          bbox_id_to_point_num, bbox_id_to_view_id, bbox_id_to_distance, sort_flag):
        #添加ts0时刻中包含box的数据
        framedata_group = f[str(ts0)]
        framedata_group.create_dataset('box_ids', data=np.array(bbox_id_mapping, dtype='S'))
        framedata_group.create_dataset('box_poses', data=np.array(bbox_id_to_pose, dtype=np.float32))
        for i, bbox_id in enumerate(bbox_id_mapping):
            if str(bbox_id) not in f.keys():
                # 初始化新Box存储结构
                box_group = f.create_group(str(bbox_id))
                box_group.attrs['first_seen'] = str(ts0)
                box_group.attrs['last_seen'] = str(ts0)

                # 创建结构化数组存储 frames
                frame_data = np.array(
                    [(bbox_id_to_point_num[i], bbox_id_to_distance[i], bbox_id_to_view_id[i], np.string_(str(ts0)))],
                    dtype=frame_dtype
                )
                box_group.create_dataset('frames', data=frame_data, maxshape=(None,))
            else:
                # 更新Box时间范围
                box_group = f[str(bbox_id)]
                box_group.attrs['last_seen'] = str(ts0)

                frames_dset = box_group['frames']
                # 扩展数据集并追加新数据
                new_frame = np.array(
                    [(bbox_id_to_point_num[i], bbox_id_to_distance[i], bbox_id_to_view_id[i], np.string_(str(ts0)))],
                    dtype=frame_dtype
                )
                frames_dset.resize((frames_dset.shape[0] + 1,))
                frames_dset[-1] = new_frame

            # #! for debug
            # if str(ts0) in f.keys():
            #     del f[str(ts0)]
            frame_group = box_group.create_group(f'{str(ts0)}')

            frame_group.create_dataset('vertices', data=bbox_id_to_vertices[i].astype(np.float32))
            frame_group.create_dataset('point', data=bbox_id_to_point[i].astype(np.float32))
            frame_group.create_dataset('pose', data=bbox_id_to_pose[i].astype(np.float32))
            frame_group.create_dataset('ego_pose', data=bbox_id_to_ego_pose[i].transform_matrix.astype(np.float32))
            frame_group.create_dataset('lwh_center', data=bbox_id_to_lwh_center[i].astype(np.float32))
  
            frame_group.attrs['category'] = bbox_id_to_category[i]
            frame_group.attrs['dynamic_status'] = bbox_id_to_dynamic_status[i]
            frame_group.attrs['valid_status'] = bbox_id_to_valid_status[i]

        if sort_flag:
            for box_id in f.keys():
                box_group = f[str(box_id)]
                if 'frames' not in box_group.keys():
                    continue
                frame_list = box_group['frames']
                select_list = []
                for frame in frame_list:
                    select_list.append((frame['point_num'], frame['distance'], frame['view_id'], frame['timestamp'].decode()))
        
                sorted_frame_list = rank_based_dynamic_sort(select_list, alpha=0.5)
                sorted_frame_list_np = list_to_np(sorted_frame_list)
                box_group.create_dataset('sorted_frame_list', data = sorted_frame_list_np)



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
        #添加ts0时刻中包含box的数据
        all_inf_group = f.create_group('all_information')
        all_inf_group.create_dataset('timestamp_list', data=np.array(timestamps, dtype='S'))
        for cnt, ts0 in enumerate(timestamps):
            # #! for debug
            # if str(ts0) in f.keys():
            #     del f[str(ts0)]
                
            # group = f.create_group(str(ts0))
            # pc0, pose0, is_ground_0 = read_pose_pc_ground(data_dir, log_id, ts0, avm)
            if cnt == len(timestamps) - 1:
                ts_ = timestamps[cnt - 1] # 最后一帧相邻帧用前一阵代替
                scene_flow = compute_sceneflow(data_dir, log_id, (ts0, ts_), avm)
                sort_flag = True
            else:
                ts1 = timestamps[cnt + 1]
                scene_flow = compute_sceneflow(data_dir, log_id, (ts0, ts1), avm)
                sort_flag = False
                # create_group_data(group, pc0, is_ground_0.astype(np.bool_), pose0.transform_matrix.astype(np.float32),
                #                   scene_flow['flow_0_1'], scene_flow['valid_0'], scene_flow['classes_0'],
                #                   scene_flow['ego_motion'].transform_matrix.astype(np.float32), 
                #                   scene_flow['classes_box'], scene_flow['bbox_id_to_vertices'], 
                #                   scene_flow['bbox_id_mapping'], scene_flow['bbox_id_to_category'])
            create_group_data(f, ts0, scene_flow['bbox_id_to_vertices'], scene_flow['bbox_id_mapping'], scene_flow['bbox_id_to_category'],
                                scene_flow['bbox_id_to_point'], 
                                scene_flow['bbox_id_to_pose'], scene_flow['bbox_id_to_ego_pose'], 
                                scene_flow['bbox_id_to_dynamic_status'], scene_flow['bbox_id_to_valid_status'],
                                scene_flow['bbox_id_to_lwh_center'],  scene_flow['bbox_id_to_point_num'], 
                                scene_flow['bbox_id_to_view_id'], scene_flow['bbox_id_to_distance'], sort_flag)

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_dir: Path, output_dir: Path, nproc: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    
    if not data_dir.exists():
        print(f'{data_dir} not found')
        return
    
    # NOTE(Qingwen): if you don't want to all data_dir, then change here: logs = logs[:10] only 10 scene.
    # logs = os.listdir(data_dir)[1:10]
    #! 追加数据
    logs = os.listdir(output_dir)
    args = sorted([(data_dir, log.rstrip('.h5'), output_dir) for log in logs])
    print(f'Using {nproc} processes to process data: {data_dir} to .h5 format. (#scenes: {len(args)})')
    # for debug
    for x in tqdm(args):
        proc(x, ignore_current_process=True)
        # break
    # if nproc <= 1:
    #     for x in tqdm(args, ncols=120):
    #         proc(x, ignore_current_process=True)
    # else:
    #     with Pool(processes=nproc) as p:
    #         res = list(tqdm(p.imap_unordered(proc, args), total=len(logs), ncols=120))

def main(
    argo_dir: str = "/data0/dataset/av2/",
    output_dir: str ="/data1/dataset/av2/debug/",
    av2_type: str = "sensor",
    data_mode: str = "val",
    mask_dir: str = "/data0/dataset/av2/eval_mask222/",
    nproc: int = 1, #(multiprocessing.cpu_count() - 1)
):
    data_root_ = Path(argo_dir) / av2_type/ data_mode
    output_dir_ = Path(output_dir) / av2_type / data_mode
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(data_root_, output_dir_, nproc)
    create_reading_mixdata_frame(output_dir_)
    # if data_mode == "val" or data_mode == "test":
    #     create_eval_mask(data_mode, output_dir_, mask_dir)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")