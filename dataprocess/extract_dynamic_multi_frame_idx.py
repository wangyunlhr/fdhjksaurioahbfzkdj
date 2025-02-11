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

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index

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
def process_log(data_dir: Path, log_id: str, output_dir: Path, multi_frame: int, n: Optional[int] = None) :

    def create_group_data(group, pc, pose, point_valid, gm, target_mask=None, flow_0to1=None, 
                          flow_valid=None, flow_category=None, ego_motion=None, class_valid=None):
        # if pc is not None:
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('ground_mask', data=gm.astype(bool))
        group.create_dataset('point_valid', data=point_valid.astype(bool))
        group.create_dataset('pose', data=pose.astype(np.float32))
        group.create_dataset('target_mask', data=target_mask.astype(bool))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
            group.create_dataset('class_valid', data=class_valid.astype(bool))

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
            print('multi_frame:', multi_frame)
            mid_frame = multi_frame // 2
            if cnt == len(timestamps) - 1:
                stage = 'last'
                scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt-multi_frame+1:], avm, stage, ts0)
                create_group_data(group, scene_flow['acc_pc0'], pose0.transform_matrix.astype(np.float32), 
                                  scene_flow['valid_point'], scene_flow['ground_point'].astype(np.bool_), scene_flow['target_mask'],)
            else:
                if cnt < mid_frame:
                    stage = 'start'
                    scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt:cnt + multi_frame], avm, stage, ts0)
                elif cnt >= len(timestamps) - mid_frame:
                    stage = 'end'
                    scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt-multi_frame+2:cnt+2], avm, stage, ts0)
                else:
                    stage ='mid'
                    scene_flow = compute_sceneflow(data_dir, log_id, timestamps[cnt-mid_frame:cnt+mid_frame+1], avm, stage, ts0)

                assert scene_flow['acc_pc0'].shape[0] == scene_flow['class_valid'].shape[0]
                create_group_data(group, scene_flow['acc_pc0'], pose0.transform_matrix.astype(np.float32), #! change to new_pc0
                                  scene_flow['valid_point'], scene_flow['ground_point'].astype(np.bool_), scene_flow['target_mask'],
                                  scene_flow['flow_0_1'], scene_flow['valid_flow_0'], scene_flow['classes_0'],
                                  scene_flow['ego_motion'].transform_matrix.astype(np.float32),
                                  scene_flow['class_valid'])

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_dir: Path, output_dir: Path, nproc: int, multi_frame: int):
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
    args = sorted([(data_dir, log, output_dir,  multi_frame) for log in logs])
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
    nproc: int = (multiprocessing.cpu_count() - 1)
):
    data_root_ = Path(argo_dir) / av2_type/ data_mode
    output_dir_ = Path(output_dir) / av2_type / data_mode
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(data_root_, output_dir_, nproc, multi_frame)
    create_reading_index(output_dir_)
    if data_mode == "val" or data_mode == "test":
        create_eval_mask(data_mode, output_dir_, mask_dir)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")