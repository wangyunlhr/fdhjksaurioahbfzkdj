"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: view scene flow dataset after preprocess.
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import flow_to_rgb 
# from src.dataset import HDF5Dataset as HDF5Data
from scripts.network.dataloader_flow4D import HDF5Dataset_onlybox_multiframe, collate_fn_pad
from src.utils.o3d_view import MyVisualizer, color_map
import yaml
import pickle
import torch


VIEW_FILE = f"{BASE_DIR}/assets/view/av2.json"

def check_flow(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow", # "flow", "flow_est"
    start_id: int = -1,
    point_size: float = 3.0,
):
    dataset = HDF5Data(data_dir, n_frames=5)

    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    opt.point_size = point_size

    for data_id in (pbar := tqdm(range(0, len(dataset)))):
        # for easy stop and jump to any id, and save same id always from 0.
        if data_id < start_id and start_id != -1:
            continue
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        gm0 = data['gm0']
        pc0 = data['pc0'][~gm0]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([1.0, 0.0, 0.0]) # red: pc0

        pc1 = data['pc1']
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3][~data['gm1']])
        pcd1.paint_uniform_color([0.0, 1.0, 0.0]) # green: pc1

        pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + pose_flow) # if you want to check pose_flow
        pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + data[res_name][~gm0])
        pcd2.paint_uniform_color([0.0, 0.0, 1.0]) # blue: pc0 + flow
        o3d_vis.update([pcd, pcd1, pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

def vis(
    data_dir: str = "/data1/dataset/av2/multi_box/sensor/train", #"/data1/dataset/av2/preprocess/sensor/train/", #"/data1/dataset/av2/multi_box/sensor/train",
    res_name: str = "flow", # "flow", "flow_est"
    start_id: int = 0,
    point_size: float = 2.0,
):  
    
    dataset = dataset = HDF5Dataset_onlybox_multiframe(data_dir, 2, eval=False)
    o3d_vis = MyVisualizer(view_file="/data0/code/Flow4D_croco/tools/logs/imgs_multi_box/ScreenView_eb222d5d-0052-3ce7-9b87-19e09054a2c0_315967979860320000.json", window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop", save_folder='./logs/imgs_mutli_box_motion')
    # label mapping
    with open('/data0/code/Flow4D_croco/conf/labeling.yaml', 'r') as file:
        labeling_map = yaml.safe_load(file)
    class_map = labeling_map['Argoverse_labels']
    opt = o3d_vis.vis.get_render_option()
    # opt.background_color = np.asarray([216, 216, 216]) / 255.0
    # opt.background_color = np.asarray([80/255, 90/255, 110/255])
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = point_size

    #check eval_mask
    last_id = 0
 
    for data_id in (pbar := tqdm(range(0, len(dataset)))):
        # for easy stop and jump to any id, and save same id always from 0.
        # from IPython import embed; embed()
        if data_id < start_id and start_id != -1:
            continue
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        if data_id > 10 and last_id != now_scene_id:
            break
        last_id = now_scene_id
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        

        # pose0 = data['pose0']
        # pose1 = data['pose1']
        # ego_pose = np.linalg.inv(pose1) @ pose0

        
        #!check motion flow
        # ego_pose = data['ego_motion']
        # pc0 = data['pc0']
        # pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]

        # flow = data['flow'].numpy()
        # flow_color = flow_to_rgb(flow) / 255.0

        # is_dynamic = torch.linalg.vector_norm(data['flow'] - pose_flow, dim=-1) >= 0.05
        # flow_color[~is_dynamic] = [1, 1, 1]

        # from IPython import embed; embed()
        # labels_color 
        # if res_name in ['dufo_label', 'label']:
        #!check classes
        # pc0 = data['pc0'].numpy()
        # pcd = o3d.geometry.PointCloud()
        # labels = data['flow_category_labeled']
        # pcd_i = o3d.geometry.PointCloud()
        # for label_i in np.unique(labels):
        #     pcd_i.points = o3d.utility.Vector3dVector(pc0[labels == label_i][:, :3])
        #     if label_i <= 0:
        #         continue
        #         pcd_i.paint_uniform_color([1.0, 1.0, 1.0])
        #     else:
        #         pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
        #         print(f"Category: {label_i}, Color: {color_map[label_i % len(color_map)]}")
        #     pcd += pcd_i

            # pbar.set_description(f"labels: {list(map(lambda num: class_map.get(num, 'unknown'), np.unique(labels)))}")
            # o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
        # check flow2
        # from IPython import embed; embed() 
        #data.keys() dict_keys(['scene_id', 'timestamp', 'pc0', 'pose0', 'pc1', 'pose1', 'flow', 'flow_is_valid', 'flow_category_indices', 'flow_category_labeled', 'ego_motion'])

        pc0 = (data['pc0']+data['flow'])#[~(data['gm0'] | (data['flow_category_indices'] == 0))]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([0.0, 1.0, 0.0]) # green
        # pcd.colors = o3d.utility.Vector3dVector(flow_color)

        pc1 = data['pc1'] #[~(data['gm1'] | (data['flow_category_indices_pc1'] == 0))]
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3])
        pcd1.paint_uniform_color([0.0, 0.0, 1.0]) # green


        o3d_vis.update([pcd,pcd1, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)], name=now_scene_id + '_'+ data['timestamp'])

        # flow_color
        # elif res_name in data:
        # pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        # flow = data['flows_adjacent_s_1'].numpy()
        # flow_color = flow_to_rgb(flow) / 255.0
        # # is_dynamic = np.linalg.norm(flow, axis=1) > 0.1
        # # flow_color[~is_dynamic] = [1, 1, 1]
        # flow_color = 0.5 * np.ones((pc0.shape[0],3))
        # flow_color[gm0] = [0, 0, 0]
        # pcd.colors = o3d.utility.Vector3dVector(flow_color)


if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(check_flow)
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")