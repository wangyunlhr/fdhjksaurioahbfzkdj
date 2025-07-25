"""
# Created: 2023-11-01 17:02
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

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
from copy import deepcopy

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index
from src.utils.av2_eval import read_ego_SE3_sensor

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
    mask_file_path = Path(mask_dir) / f"{data_mode}-masks.zip"
    if not mask_file_path.exists():
        print(f'{mask_file_path} not found, please download the mask file for official evaluation.')
        return
    # extract the mask file
    # with ZipFile(mask_file_path, 'r') as zipObj:
    #     zipObj.extractall(Path(mask_dir) / f"{data_mode}-masks")
    
    data_index = []
    # list scene ids
    scene_ids = os.listdir(Path(mask_dir) / f"{data_mode}-masks")
    for scene_id in tqdm(scene_ids, desc=f'Create {data_mode} eval mask', ncols=100):
        timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(Path(mask_dir) / f"{data_mode}-masks" / scene_id)
                        if file.endswith('.feather')])
        if not os.path.exists(output_dir_ / f'{scene_id}.h5'):
            continue
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
    # more detail: https://argoverse.github.io/user-guide/datasets/lidar.html#sensor-suite
    ego2sensor_pose = read_ego_SE3_sensor((data_dir / log_id))['up_lidar']
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin([timestamp])]
    pose = convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == timestamp])
    pc = Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{timestamp}.feather").xyz
    # transform to city coordinate since sweeps[0].xyz is in ego coordinate to get ground mask
    is_ground = avm.get_ground_points_boolean(pose.transform_point_cloud(pc))

    # NOTE(SeFlow): transform to sensor coordinate, since some ray-casting based methods need sensor coordinate
    pc = ego2sensor_pose.inverse().transform_point_cloud(pc) 
    return pc, pose, is_ground





def compute_sceneflow_one(data_dir: Path, log_id: str, timestamps: Tuple[int, int], avm: ArgoverseStaticMap) -> Dict[str, Union[np.ndarray, SE3]]:
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
    def compute_flow_one(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        
        flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        # Convert to float32s
        flow = flow.astype(np.float32)
        #!
        pose_flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        pose_flow = pose_flow.astype(np.float32)
        source_point = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz)


        valid = np.ones(len(sweeps[0].xyz), dtype=np.bool_)
        # classes = -np.ones(len(sweeps[0].xyz), dtype=np.int8)
        classes = np.zeros(len(sweeps[0].xyz), dtype=np.uint8)
        classes_1 = np.zeros(len(sweeps[1].xyz), dtype=np.uint8)

        # # old version
        # for id in cuboids[0]:
        #     c0 = cuboids[0][id]
        #     c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
        #     c0.width_m += BOUNDING_BOX_EXPANSION
        #     obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
        #     classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
        
        #     if id in cuboids[1]:
        #         c1 = cuboids[1][id]
        #         c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
        #         obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
        #         flow[obj_mask] = obj_flow.astype(np.float32)
        #     else:
        #         valid[obj_mask] = 0

        # NOTE(HiMo): box expansion based on the object velocity
        # check more detail: https://kin-zhang.github.io/HiMo
        for id in cuboids[0]:
            c0 = deepcopy(cuboids[0][id])
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse()))
                rel_obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
                delta_move = abs(np.linalg.norm(rel_obj_flow, axis=0).mean())

                if delta_move > 0.04: # only when it's moving
                    c0 = cuboids[0][id]
                    c0.length_m += (BOUNDING_BOX_EXPANSION + min(delta_move/2, 2)) # since 180/360 for two LiDARs orientation
                    c0.width_m += BOUNDING_BOX_EXPANSION
                    c0.height_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)

                # NOTE(Qingwen): after expansion, we need to recompute the flow
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
                flow[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid[obj_mask] = 0
        flow = flow-pose_flow

        for id in cuboids[1]:
            c_11 = deepcopy(cuboids[1][id])
            obj_pts1, obj_mask1 = c_11.compute_interior_points(sweeps[1].xyz)
            classes_1[obj_mask1] = CATEGORY_TO_INDEX[str(c_11.category)]

        return flow, classes, valid, ego1_SE3_ego0, source_point, classes_1
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
    ground_masks = [avm.get_ground_points_boolean(poses[i].transform_point_cloud(sweeps[i].xyz)) for i in range(len(sweeps))]

    flow_0_1, classes_0, valid_0, ego_motion, source_point_0, classes_1 = compute_flow_one(sweeps, cuboids, poses)

    return {'pcl_0': source_point_0, 'pcl_1' :sweeps[1].xyz, 'flow_0_1': flow_0_1,
            'valid_0': valid_0, 'classes_0': classes_0,  'classes_1': classes_1,
            'pose_0': poses[0], 'pose_1': poses[1],
            'ego_motion': ego_motion, 'ground_mask_0': ground_masks[0], 'ground_mask_1': ground_masks[1]}






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
        
        flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        # Convert to float32s
        flow = flow.astype(np.float32)
        #! add poseflow
        pose_flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        pose_flow = pose_flow.astype(np.float32)
        source_point = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz)

        valid = np.ones(len(sweeps[0].xyz), dtype=np.bool_)
        # classes = -np.ones(len(sweeps[0].xyz), dtype=np.int8)
        classes = np.zeros(len(sweeps[0].xyz), dtype=np.uint8)

        # # old version
        # for id in cuboids[0]:
        #     c0 = cuboids[0][id]
        #     c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
        #     c0.width_m += BOUNDING_BOX_EXPANSION
        #     obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
        #     classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
        
        #     if id in cuboids[1]:
        #         c1 = cuboids[1][id]
        #         c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
        #         obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
        #         flow[obj_mask] = obj_flow.astype(np.float32)
        #     else:
        #         valid[obj_mask] = 0

        # NOTE(HiMo): box expansion based on the object velocity
        # check more detail: https://kin-zhang.github.io/HiMo
        for id in cuboids[0]:
            c0 = deepcopy(cuboids[0][id])
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse()))
                rel_obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
                delta_move = abs(np.linalg.norm(rel_obj_flow, axis=0).mean())

                if delta_move > 0.04: # only when it's moving
                    c0 = cuboids[0][id]
                    c0.length_m += (BOUNDING_BOX_EXPANSION + min(delta_move/2, 2)) # since 180/360 for two LiDARs orientation
                    c0.width_m += BOUNDING_BOX_EXPANSION
                    c0.height_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)

                # NOTE(Qingwen): after expansion, we need to recompute the flow
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
                flow[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid[obj_mask] = 0
        flow = flow-pose_flow
        
        return flow, classes, valid, ego1_SE3_ego0, source_point
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
    ground_masks = [avm.get_ground_points_boolean(poses[i].transform_point_cloud(sweeps[i].xyz)) for i in range(len(sweeps))]

    flow_0, classes_0, valid_0, ego_motion_0, point0_intarget = compute_flow([sweeps[0],sweeps[4]], [cuboids[0],cuboids[4]], [poses[0],poses[4]]) #flow去除了egomotion, point转换到目标帧
    flow_1, classes_1, valid_1, ego_motion_1, point1_intarget = compute_flow([sweeps[1],sweeps[4]], [cuboids[1],cuboids[4]], [poses[1],poses[4]]) 
    flow_2, classes_2, valid_2, ego_motion_2, point2_intarget = compute_flow([sweeps[2],sweeps[4]], [cuboids[2],cuboids[4]], [poses[2],poses[4]]) 
    flow_3, classes_3, valid_3, ego_motion_3, point3_intarget = compute_flow([sweeps[3],sweeps[4]], [cuboids[3],cuboids[4]], [poses[3],poses[4]]) 

    
    return {'pcl_0': point0_intarget, 'pcl_1': point1_intarget, 'pcl_2': point2_intarget, 'pcl_3': point3_intarget, 'pcl_4': sweeps[4].xyz,
            'flow_0': flow_0, 'flow_1': flow_1, 'flow_3': flow_3, 'flow_2': flow_2,
            'valid_0': valid_0, 'valid_1': valid_1, 'valid_3': valid_3, 'valid_2': valid_2,
            'classes_0': classes_0, 'classes_1': classes_1, 'classes_3': classes_3, 'classes_2': classes_2,
            'ego_motion_0': ego_motion_0, 'ego_motion_1': ego_motion_1, 'ego_motion_3': ego_motion_3, 'ego_motion_2': ego_motion_2,
            'ground_mask_0': ground_masks[0], 'ground_mask_1': ground_masks[1], 'ground_mask_2': ground_masks[2], 'ground_mask_3': ground_masks[3], 'ground_mask_4': ground_masks[4]}

    # return {'pcl_0': sweeps[0].xyz, 'pcl_1' :sweeps[1].xyz, 'flow_0_1': flow_0_1,
    #         'valid_0': valid_0, 'classes_0': classes_0, 
    #         'pose_0': poses[0], 'pose_1': poses[1],
    #         'ego_motion': ego_motion}

def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None) :

    def create_group_data(group, pcl_0, pcl_1, pcl_2, pcl_3, pcl_4, 
                        flow_0, flow_1, flow_3, flow_2, 
                        valid_0, valid_1, valid_3, valid_2, 
                        classes_0, classes_1, classes_3, classes_2, 
                        ego_motion_0, ego_motion_1, ego_motion_3, ego_motion_2, 
                        ground_mask_0, ground_mask_1, ground_mask_2, ground_mask_3, ground_mask_4):

        group.create_dataset('pcl_0', data=pcl_0.astype(np.float32))
        group.create_dataset('pcl_1', data=pcl_1.astype(np.float32))
        group.create_dataset('pcl_2', data=pcl_2.astype(np.float32))
        group.create_dataset('pcl_3', data=pcl_3.astype(np.float32))
        group.create_dataset('pcl_4', data=pcl_4.astype(np.float32))
        group.create_dataset('flow_0', data=flow_0.astype(np.float32))
        group.create_dataset('flow_1', data=flow_1.astype(np.float32))
        group.create_dataset('flow_3', data=flow_3.astype(np.float32))
        group.create_dataset('flow_2', data=flow_2.astype(np.float32))
        group.create_dataset('valid_0', data=valid_0.astype(bool))
        group.create_dataset('valid_1', data=valid_1.astype(bool))
        group.create_dataset('valid_3', data=valid_3.astype(bool))
        group.create_dataset('valid_2', data=valid_2.astype(bool))
        group.create_dataset('classes_0', data=classes_0.astype(np.uint8))
        group.create_dataset('classes_1', data=classes_1.astype(np.uint8))
        group.create_dataset('classes_3', data=classes_3.astype(np.uint8))
        group.create_dataset('classes_2', data=classes_2.astype(np.uint8))
        group.create_dataset('ego_motion_0', data=ego_motion_0.astype(np.float32))
        group.create_dataset('ego_motion_1', data=ego_motion_1.astype(np.float32))
        group.create_dataset('ego_motion_3', data=ego_motion_3.astype(np.float32))
        group.create_dataset('ego_motion_2', data=ego_motion_2.astype(np.float32))
        group.create_dataset('ground_mask_0', data=ground_mask_0.astype(bool))
        group.create_dataset('ground_mask_1', data=ground_mask_1.astype(bool))
        group.create_dataset('ground_mask_2', data=ground_mask_2.astype(bool))
        group.create_dataset('ground_mask_3', data=ground_mask_3.astype(bool))
        group.create_dataset('ground_mask_4', data=ground_mask_4.astype(bool))



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

    gt_flow_flag = False if not (data_dir / log_id / "annotations.feather").exists() else True

    # if n is not None:
    #     iter_bar = tqdm(zip(timestamps, timestamps[1:]), leave=False,
    #                      total=len(timestamps) - 1, position=n,
    #                      desc=f'Log {log_id}')
    # else:
    #     iter_bar = zip(timestamps, timestamps[1:])

    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, ts0 in enumerate(timestamps):
            # print('cnt', cnt)
            group = f.create_group(str(ts0))
            pc0, pose0, is_ground_0 = read_pose_pc_ground(data_dir, log_id, ts0, avm)
            if pc0.shape[0] < 256:
                print(f'{log_id}/{ts0} has less than 256 points, skip this scenarios. Please check the data if needed.')
                break
            if cnt == len(timestamps) - 1 or not gt_flow_flag:
                continue
            elif cnt < 3:
                ts1 = timestamps[cnt + 1]
                scene_flow = compute_sceneflow_one(data_dir, log_id, (ts0, ts1), avm)
                create_group_data(group, 
                                  scene_flow['pcl_0'], scene_flow['pcl_0'], scene_flow['pcl_0'], scene_flow['pcl_0'], scene_flow['pcl_1'],
                                  scene_flow['flow_0_1'], scene_flow['flow_0_1'], scene_flow['flow_0_1'], scene_flow['flow_0_1'],
                                  scene_flow['valid_0'], scene_flow['valid_0'], scene_flow['valid_0'], scene_flow['valid_0'],
                                  scene_flow['classes_0'], scene_flow['classes_0'], scene_flow['classes_0'], scene_flow['classes_0'],
                                  scene_flow['ego_motion'].transform_matrix.astype(np.float32), scene_flow['ego_motion'].transform_matrix.astype(np.float32),
                                  scene_flow['ego_motion'].transform_matrix.astype(np.float32), scene_flow['ego_motion'].transform_matrix.astype(np.float32),
                                  scene_flow['ground_mask_0'].astype(np.bool_), scene_flow['ground_mask_0'].astype(np.bool_), 
                                  scene_flow['ground_mask_0'].astype(np.bool_), scene_flow['ground_mask_0'].astype(np.bool_), scene_flow['ground_mask_1'].astype(np.bool_))

            else:
                ts1 = timestamps[cnt - 3]
                ts2 = timestamps[cnt - 2]
                ts3 = timestamps[cnt - 1]
                ts4 = timestamps[cnt + 1]
                scene_flow = compute_sceneflow(data_dir, log_id, (ts1, ts2, ts3, ts0, ts4), avm)
                create_group_data(group, 
                                  scene_flow['pcl_0'], scene_flow['pcl_1'], scene_flow['pcl_2'], scene_flow['pcl_3'], scene_flow['pcl_4'],
                                  scene_flow['flow_0'], scene_flow['flow_1'], scene_flow['flow_3'], scene_flow['flow_2'],
                                  scene_flow['valid_0'], scene_flow['valid_1'], scene_flow['valid_3'], scene_flow['valid_2'],
                                  scene_flow['classes_0'], scene_flow['classes_1'], scene_flow['classes_3'], scene_flow['classes_2'],
                                  scene_flow['ego_motion_0'].transform_matrix.astype(np.float32), scene_flow['ego_motion_1'].transform_matrix.astype(np.float32),
                                  scene_flow['ego_motion_3'].transform_matrix.astype(np.float32), scene_flow['ego_motion_2'].transform_matrix.astype(np.float32),
                                  scene_flow['ground_mask_0'].astype(np.bool_), scene_flow['ground_mask_1'].astype(np.bool_), 
                                  scene_flow['ground_mask_2'].astype(np.bool_), scene_flow['ground_mask_3'].astype(np.bool_), scene_flow['ground_mask_4'].astype(np.bool_))
                                 
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
    logs = os.listdir(data_dir)
    args = sorted([(data_dir, log, output_dir) for log in logs])
    print(f'Using {nproc} processes to process data: {data_dir} to .h5 format. (#scenes: {len(args)})')
    # for debug
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
    argo_dir: str = "/data0/dataset/av2/",
    output_dir: str ="/data1/dataset/av2/preprocess_lidiff_past",
    av2_type: str = "sensor",
    data_mode: str = "val",
    mask_dir: str = "/data0/dataset/av2/eval_mask/",
    nproc: int = (multiprocessing.cpu_count() - 1),
    only_index: bool = True,
):
    data_root_ = Path(argo_dir) / av2_type/ data_mode
    output_dir_ = Path(output_dir) / av2_type / data_mode
    if only_index:
        create_reading_index(output_dir_)
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(data_root_, output_dir_, nproc)
    create_reading_index(output_dir_)
    if data_mode == "val" or data_mode == "test":
        create_eval_mask(data_mode, output_dir_, mask_dir)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")
