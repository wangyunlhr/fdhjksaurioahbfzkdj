"""
# Created: 2023-11-04 15:52
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Torch dataloader for the dataset we preprocessed.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import Dataset, DataLoader
import h5py, os, pickle, argparse
from tqdm import tqdm
import yaml
import numpy as np

# from models.basic.make_voxels import HardVoxelizer, DynamicVoxelizer
# import multiprocessing
# lock = multiprocessing.Lock()



def collate_fn_pad_track(batch):

    # num_frames = 2
    # while f'pc_m{num_frames - 1}' in batch[0]:
    #     num_frames += 1

    # padding the data
    pc0_ori, pc1_ori= [], []
    pc0_gm0, pc1_gm1= [], []

    # pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_ori.append(batch[i]['pc0'])
        pc1_ori.append(batch[i]['pc1'])
        pc0_gm0.append(batch[i]['gm0'])
        pc1_gm1.append(batch[i]['gm1'])
        # for j in range(1, num_frames - 1):
            # pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_ori = torch.nn.utils.rnn.pad_sequence(pc0_ori, batch_first=True, padding_value=torch.nan)
    pc1_ori = torch.nn.utils.rnn.pad_sequence(pc1_ori, batch_first=True, padding_value=torch.nan)
    pc0_gm0 = torch.nn.utils.rnn.pad_sequence(pc0_gm0, batch_first=True, padding_value=torch.nan)
    pc1_gm1 = torch.nn.utils.rnn.pad_sequence(pc1_gm1, batch_first=True, padding_value=torch.nan)
    # pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0': pc0_ori,
        'pc1': pc1_ori,
        'gm0': pc0_gm0,
        'gm1': pc1_gm1,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
    }

    # for j in range(1, num_frames - 1):
    #     res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
    #     res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]: #NOTE 全部都没有去除地面点
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    #NOTE 新增box序列处理

    box_data = {
        'batch_indices': [],        # 记录每个box所属的batch索引
        'original_box_idx': [],     # 记录每个box在原始点云中的索引
        'box_pc0': [],             # 当前帧box点云（归一化坐标系）
        'box_p0_dense': [],        # 时序点云序列
        'box_pc1': []              # 下一帧box点云（归一化坐标系）
    }

    # 遍历batch收集box数据
    for batch_idx, sample in enumerate(batch):
        if 'frame_boxes' in sample:
            for box in sample['frame_boxes']:
                box_data['batch_indices'].append(batch_idx)
                box_data['original_box_idx'].append(box['box_idx'])
                
                # 转换到float32并添加序列维度
                box_data['box_pc0'].append(box['box_pc0'].to(torch.float32)) 
                box_data['box_p0_dense'].append(box['box_p0_dense'].to(torch.float32))
                box_data['box_pc1'].append(box['box_pc1'].to(torch.float32))

    # 动态目标数据填充
    if box_data['box_pc0']:
        # 点云维度填充 (B*N_boxes, T, 3)
        box_data['box_pc0'] = torch.nn.utils.rnn.pad_sequence(
            [x for x in box_data['box_pc0']], 
            batch_first=True, 
            padding_value=torch.nan
        )
        
        # 时序点云填充 (B*N_boxes, num_frames, 3)
        box_data['box_p0_dense'] = torch.nn.utils.rnn.pad_sequence(
            [x for x in box_data['box_p0_dense']], 
            batch_first=True, 
            padding_value=torch.nan
        )
        
        box_data['box_pc1'] = torch.nn.utils.rnn.pad_sequence(
            [x for x in box_data['box_pc1']], 
            batch_first=True, 
            padding_value=torch.nan
        )
        

    return res_dict





#! for track_id
class HDF5Dataset_track_data(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_track_data, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('./conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        # ! eval_all image
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join('/data0/code/Flow4D_ori/assets/docs/index_eval_v2.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        #! eval all
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                index_ = index_ - 1
        scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        # with h5py.File(os.path.join('/data1/dataset/av2/box_visual/sensor/val/', f'{scene_id}.h5'), 'r') as f: 
        #     pc0_box = torch.tensor(f[key]['lidar'][:]) 
        #     gm0_box = torch.tensor(f[key]['ground_mask'][:]) 
        #     pose0_box = torch.tensor(f[key]['pose'][:]) 

        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            #! 1.获取全部的box_id
            box_ids0 = f[key]['box_ids'][:]
            box_poses0 = f[key]['box_poses'][:]
            box_category0 = f[key]['box_category'][:]
            box_mask0 = f[key]['box_mask'][:]

            pc0 = torch.tensor(f[key]['lidar'][:]) 
            gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 
            # assert torch.equal(pc0_box, pc0)
            # assert torch.equal(gm0_box, gm0)
            # assert torch.equal(pose0_box, pose0)

            if self.scene_id_bounds[scene_id]["max_index"] == index_:
                print("!!!!!!__getitem__(index_ + 1)") 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])
            pose1_to_0 = cal_pose0to1(pose1, pose0) #! 这里是ego_pose
            pose0_to_1 = cal_pose0to1(pose0, pose1)

            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  
                frame_boxes = []
                for i, box_id in enumerate(box_ids0): #对每一个box进行筛选
                    #! 1先判断下一帧有没有同款
                    if box_id not in f[next_timestamp]['box_ids'][:]:
                        continue
                    past_frames_for_box0 = []
                    past_frames_for_box1 = []
                    #! 存储全局中box的映射关系
                    box_idx = np.where(box_mask0 == box_id)[0]
                    box_pc0 = pc0[box_idx]

                    box_group = f[str(box_id)]
                    #! 获取当前帧box的pose
                    box_pose0 = torch.tensor(box_group[key]['pose'][:])
                    box_pose1 = torch.tensor(box_group[next_timestamp]['pose'][:])
                    box_pose0_inv = cal_pose_inv(box_pose0)
                    transformed_box_pc0 = box_pc0 @ box_pose0_inv[:3, :3].T + box_pose0_inv[:3, 3] #将box点从当前帧坐标系转换到box0坐标系下
                    select_best = box_group['sorted_frame_list']
                    select_best_sorted = split_frame(select_best, num_past_frames)

                    #NOTE 用于debug flow
                    flow = torch.tensor(f[key]['flow'][:])
                    flow_box = flow[box_idx]
                    ego_flow_box = box_pc0 @ pose0_to_1[:3, :3].T + pose0_to_1[:3, 3] - box_pc0 
                    flow_gt_box = ((box_pc0 + flow_box) @ pose1_to_0[:3, :3].T + pose1_to_0[:3, 3]) @ box_pose0_inv[:3, :3].T + box_pose0_inv[:3, 3] - transformed_box_pc0 #! 这里是flow_gt
                    #! box_flow转换成原始已经补偿了ego_motion的flow
                    flow_restore = ((transformed_box_pc0.to(torch.float32) + flow_gt_box.to(torch.float32)) @ box_pose0[:3, :3].T + box_pose0[:3, 3]) @ pose0_to_1[:3, :3].T + pose0_to_1[:3, 3] - (box_pc0@ pose0_to_1[:3, :3].T + pose0_to_1[:3, 3])

                    #! 下一时刻的同一个box_id的点————>转换到0帧坐标系下
                    box_pc1 = torch.tensor(box_group[next_timestamp]['point'][:])
                    box_pc1_frame0 = (box_pc1 @ pose1_to_0[:3, :3].T + pose1_to_0[:3, 3]).to(torch.float32)
                    box_pc1_frame0_inbox0 = (box_pc1_frame0 @ box_pose0_inv[:3, :3].T + box_pose0_inv[:3, 3]).to(torch.float32)
                    # select_best_sorted += [key, next_timestamp]

                    for j, best_frame in enumerate(select_best_sorted): #对每一个最佳帧进行筛选
                        point = box_group[best_frame]['point']
                        pose = box_group[best_frame]['pose']
                        pose_inv = cal_pose_inv(pose)
                        transformed_point = point @ pose_inv[:3, :3].T + pose_inv[:3, 3]
                        past_pc = torch.tensor(transformed_point, dtype=torch.float32)
                        past_for_one = ((past_pc @ box_pose1[:3, :3].T + box_pose1[:3, 3]) @ pose1_to_0[:3, :3].T + pose1_to_0[:3, 3]) @ box_pose0_inv[:3, :3].T + box_pose0_inv[:3, 3]
                        
                        # if True:
                        #     pose_noise = add_se3_noise(np.eye(4))
                        #     transformed_point = transformed_point @ pose_noise[:3, :3].T + pose_noise[:3, 3]
                        # if box_group[best_frame].attrs['category'] == 19:
                        #     os.makedirs(f"./for_visual", exist_ok=True)
                        #     save_lidar(point, f"./for_visual/{key}_{box_id}_{j}_point.ply")
                        #     save_lidar(transformed_point, f"./for_visual/{key}_{box_id}_{j}_transformed_point.ply")
                        #     print( f"./for_visual/{box_id}_{j}.ply")
                        past_frames_for_box0.append(past_pc)
                        past_frames_for_box1.append(past_for_one)
                    past_frames_for_box0.append(transformed_box_pc0)
                    past_frames_for_box1.append(box_pc1_frame0_inbox0)
                    box_data = {
                        'box_idx': box_idx, #对应pc0的索引 (N,)
                        'box_pc0': transformed_box_pc0, # (N,3)
                        'box_p0_dense': torch.cat(past_frames_for_box0).to(torch.float32), # (M1,3)
                        'box_pc1': torch.cat(past_frames_for_box1).to(torch.float32), # (M2,3)
                    }
                    frame_boxes.append(box_data)

                    # if box_group[key].attrs['category'] == 19 and transformed_box_pc0.shape[0] > 500 and box_group[key].attrs['dynamic_status']:
                    #     os.makedirs(f"./for_visual", exist_ok=True)
                    #     save_lidar(box_pc0.cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc0.ply")
                    #     save_lidar(box_pc1.cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc1.ply")
                    #     save_lidar((box_pc0 + ego_flow_box + flow_restore).cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc0_flow.ply")
                    #     # save_lidar(box_pc1_frame0_inbox0.cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc1_frame0_inbox0.ply")
                    #     save_lidar(torch.cat(past_frames_for_box0).cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc0_dense.ply")
                    #     # save_lidar(transformed_box_pc0.cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc0.ply")
                    #     save_lidar(torch.cat(past_frames_for_box1).to(torch.float32).cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc1_dense.ply")
                    #     save_lidar((transformed_box_pc0 + flow_gt_box).cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc0_dense_flow_gt.ply")
                    #     # save_lidar(box_pc1_frame0_inbox0.cpu().numpy(), f"./for_visual/{key}_{box_id}_{j}_box_pc1.ply")

                #!
                # for i in range(1, num_past_frames + 1):
                #     frame_index = index_ - i
                #     if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                #         frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                #     past_timestamp = str(self.data_index[frame_index][1])
                #     past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                #     past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                #     past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                #     past_frames.append((past_pc, past_gm, past_pose))

                # for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                #     res_dict[f'pc_m{i+1}'] = past_pc
                #     res_dict[f'gm_m{i+1}'] = past_gm
                #     res_dict[f'pose_m{i+1}'] = past_pose
            res_dict['frame_boxes'] = frame_boxes #! 存储当前帧的box信息
            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices #原始的category属性
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor #映射之后的label

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

            #! eval all
            # res_dict['eval_mask'] = (~(gm0 | (torch.tensor(f[key]['flow_category_indices'][:]) == 0)))
            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")
            # save_flag = True
            # if save_flag:
            #     os.makedirs(f"./check_eval_mask", exist_ok=True)
            #     save_lidar(pc0.cpu().numpy(), f"./check_eval_mask/{scene_id}_{timestamp}_pc0.ply")
            #     save_lidar(pc0[eval_mask[:,0]].cpu().numpy(), f"./check_eval_mask/{scene_id}_{timestamp}_pc0_mask.ply")
                
        return res_dict









def collate_fn_pad(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground= [], []
    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['pc0'][~batch[i]['gm0']])
        pc1_after_mask_ground.append(batch[i]['pc1'][~batch[i]['gm1']])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict


def map_label(label, mapdict):
# put label from original values to xentropy
# or vice-versa, depending on dictionary values
# make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
            nel = 1
        if key > maxkey:
            maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
        lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
        lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
        try:
            lut[key] = data
        except IndexError:
            print("Wrong key ", key)
    # do the mapping
    return lut[label]




class HDF5Dataset(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('/data0/code/Flow4D_diff_less_to_more/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                index_ = index_ - 1
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            pc0 = torch.tensor(f[key]['lidar'][:]) 
            gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 

            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'time_stamps': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict



# diff完成稀疏到稠密的生成
def collate_fn_pad_less_to_more(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_all_after_mask_ground, pc1_all_after_mask_ground= [], []
    pc0_one_after_mask_ground, pc1_one_after_mask_ground= [], []
    pc0_one_mask_list = []
    time_list = []
    scene_id_list = []

    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_valid_mask = (~batch[i]['gm0'] & batch[i]['pc0_valid'])
        pc1_valid_mask = (~batch[i]['gm1'] & batch[i]['pc1_valid'])
        pc0_one_mask = (pc0_valid_mask & batch[i]['target_mask_pc0'])
        pc1_one_mask = (pc1_valid_mask & batch[i]['target_mask_pc1'])
        pc0_one_mask_list.append(pc0_one_mask)
        pc0_all_after_mask_ground.append(batch[i]['pc0'][pc0_valid_mask])
        pc1_all_after_mask_ground.append(batch[i]['pc1'][pc1_valid_mask])
        pc0_one_after_mask_ground.append(batch[i]['pc0'][pc0_one_mask])
        pc1_one_after_mask_ground.append(batch[i]['pc1'][pc1_one_mask])
        time_list.append(batch[i]['timestamp'])
        scene_id_list.append(batch[i]['scene_id'])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_all_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_all_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_all_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_all_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc0_one_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_one_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_one_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_one_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0_all': pc0_all_after_mask_ground,
        'pc1_all': pc1_all_after_mask_ground,
        'pc0': pc0_one_after_mask_ground,
        'pc1': pc1_one_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
        'timestamps': time_list,
        'scene_ids': scene_id_list,
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True) #! 不需要class mask，因为是目标帧的输出
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict



# diff完成稀疏到稠密的生成
def collate_fn_pad_NK(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_all_after_mask_ground, pc1_all_after_mask_ground= [], []
    pc0_one_after_mask_ground, pc1_one_after_mask_ground= [], []
    pc0_one_mask_list = []
    time_list = []
    scene_id_list = []

    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_one_mask_list.append(batch[i]['pc0_all_to_sparse_idx'].to(torch.long))
        pc0_all_after_mask_ground.append(batch[i]['pc0_neighbor'])
        pc1_all_after_mask_ground.append(batch[i]['pc1_neighbor'])
        pc0_one_after_mask_ground.append(batch[i]['pc0'])
        pc1_one_after_mask_ground.append(batch[i]['pc1'])
        time_list.append(batch[i]['timestamp'])
        scene_id_list.append(batch[i]['scene_id'])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_all_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_all_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_all_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_all_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc0_one_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_one_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_one_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_one_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0_all': pc0_all_after_mask_ground,
        'pc1_all': pc1_all_after_mask_ground,
        'pc0': pc0_one_after_mask_ground,
        'pc1': pc1_one_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
        'timestamps': time_list,
        'scene_ids': scene_id_list,
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True) #! 不需要class mask，因为是目标帧的输出
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][pc0_one_mask_list[i]] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict






#! multi_frame_idx 多帧加载监督M*3个点，配合collate_fn_pad_less_to_more
class HDF5Dataset_multi_frame_idx(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_multi_frame_idx, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
        # self.data_index = sorted(self.data_index) # sorted for debug

        with open('/data0/code/Flow4D_diff_less_to_more/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if (self.scene_id_bounds[scene_id]["max_index"] - 1) <= index_: #! 因为最后一阵没有target_mask
                index_ = index_ - 2
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            pc0 = torch.tensor(f[key]['lidar'][:]) 
            gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 
            pc0_valid = torch.tensor(f[key]['point_valid'][:]) 

            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])
            pc1_valid = torch.tensor(f[next_timestamp]['point_valid'][:]) 


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc0_valid': pc0_valid, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
                'pc1_valid': pc1_valid, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                target_mask_pc0 = torch.tensor(f[key]['target_mask'][:]) 
                target_mask_pc1 = torch.tensor(f[next_timestamp]['target_mask'][:]) #! pc1_mask
                class_valid = torch.tensor(f[key]['class_valid'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices
                res_dict['target_mask_pc0'] = target_mask_pc0
                res_dict['target_mask_pc1'] = target_mask_pc1
                res_dict['class_valid'] = class_valid
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict




#! multi_frame_idx 多帧加载监督N,K,3个点，配合collate_fn_pad_less_to_more_NK
class HDF5Dataset_multi_frame_idx_NK(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_multi_frame_idx_NK, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
        # self.data_index = sorted(self.data_index) # sorted for debug

        with open('/data0/code/Flow4D_diff_less_to_more/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if (self.scene_id_bounds[scene_id]["max_index"] - 1) <= index_: #! 因为最后一阵没有target_mask
                index_ = index_ - 2
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 

            # pc0 = torch.tensor(f[key]['lidar'][:]) 
            # gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 
            # pc0_valid = torch.tensor(f[key]['point_valid'][:]) 
            pc0_new = torch.tensor(f[key]['new_lidar_sparse'][:])
            pc0_new_neighbor = torch.tensor(f[key]['new_lidar_neighbor_sparse'][:])
            pc0_all_to_sparse_idx = torch.tensor(f[key]['all_to_new_lidar_sparse_idx'][:])

            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            # pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            # gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])
            # pc1_valid = torch.tensor(f[next_timestamp]['point_valid'][:]) 
            pc1_new = torch.tensor(f[next_timestamp]['new_lidar_sparse'][:])
            pc1_new_neighbor = torch.tensor(f[next_timestamp]['new_lidar_neighbor_sparse'][:])
            # pc1_all_to_sparse_idx = torch.tensor(f[next_timestamp]['all_to_new_lidar_sparse_idx'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0_new, #current
                'pc0_neighbor': pc0_new_neighbor, #current
                'pc0_all_to_sparse_idx': pc0_all_to_sparse_idx, #current
                # 'gm0': gm0, #current
                'pose0': pose0, #current
                # 'pc0_valid': pc0_valid, #current
                'pc1': pc1_new, #nect
                'pc1_neighbor': pc1_new_neighbor, #nect
                # 'gm1': gm1, #next
                'pose1': pose1, #next
                # 'pc1_valid': pc1_valid, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                target_mask_pc0 = torch.tensor(f[key]['target_mask'][:]) 
                target_mask_pc1 = torch.tensor(f[next_timestamp]['target_mask'][:]) #! pc1_mask
                class_valid = torch.tensor(f[key]['class_valid'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices
                res_dict['target_mask_pc0'] = target_mask_pc0
                res_dict['target_mask_pc1'] = target_mask_pc1
                res_dict['class_valid'] = class_valid
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict









#! change ori H5 dataset to N,K,3 
#! 需要修改voxel_size, 以及voxel_space_range
class HDF5Dataset_multi_frame_idx_changedataloader(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_multi_frame_idx_changedataloader, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
        self.data_index = sorted(self.data_index)[5990:12000] # sorted for debug
        print("load data_index[5990:12000]")

        with open('/data0/code/Flow4D_diff_less_to_more/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx
        voxel_size = [0.2, 0.2, 0.2]
        point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2]
        grid_feature_size = [512, 512, 32]
        voxel_size_4 = [v*4 for v in voxel_size]
        self.voxelizer_4 = DynamicVoxelizer(voxel_size=voxel_size_4,
                        point_cloud_range=point_cloud_range)
        self.voxel_spatial_shape_4 = [int(v/4) for v in grid_feature_size]


    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
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
    
    def __getitem__(self, index_):
        with lock:
            if self.eval_index:
                scene_id, timestamp = self.eval_data_index[index_]
                # find this one index in the total index
                index_ = self.data_index.index([scene_id, timestamp]) 
            else:
                scene_id, timestamp = self.data_index[index_] 
                # to make sure we have continuous frames
                if self.scene_id_bounds[scene_id]["max_index"] == index_: #! 因为最后一帧没有target_mask
                    index_ = index_ - 1
                scene_id, timestamp = self.data_index[index_] 

            key = str(timestamp)
            with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'a') as f: 
                # check 数据集完整性
                # for name in ['new_lidar_sparse', 'new_lidar_neighbor_sparse', 'all_to_new_lidar_sparse_idx']:
                #     if name not in f[key]:
                #         print(f"No {name} in {scene_id}.h5/{key}")
                pc0 = torch.tensor(f[key]['lidar'][:]) 
                gm0 = torch.tensor(f[key]['ground_mask'][:]) 
                pc0_valid = torch.tensor(f[key]['point_valid'][:]) 

                res_dict = {
                }
                #! 只用修改单帧
                pc0_valid_mask = (~gm0 & pc0_valid) #有效的pc0_all mask
                target_mask_pc0 = torch.tensor(f[key]['target_mask'][:]) 
                pc0_one_mask = (pc0_valid_mask & target_mask_pc0) #有效的pc0_one mask
                pc0_all = pc0[pc0_valid_mask][None].cuda()
                pc0_one = pc0[pc0_one_mask][None].cuda()

                voxel_info_dict_all = self.voxelizer_4(pc0_all)
                voxel_info_dict_all_sparse = self.voxelizer_4(pc0_one)
                #
                points_all = voxel_info_dict_all[0]['points']
                coordinates_all = voxel_info_dict_all[0]['voxel_coords'] #! z,y,x
                #
                points_sparse = voxel_info_dict_all_sparse[0]['points']
                coordinates_sparse = voxel_info_dict_all_sparse[0]['voxel_coords'] #! z,y,x
                point_idxes_sparse = voxel_info_dict_all_sparse[0]['point_idxes']
                neighbor = self.fast_knn_gpu(points_sparse, coordinates_sparse, points_all, coordinates_all, 5) # N,K,3
                #存储全部点到point_idxes_sparse的映射
                indices_A_to_B = pc0_one_mask.nonzero(as_tuple=True)[0]
                indices_A_to_C = indices_A_to_B[point_idxes_sparse.cpu()]
            #     assert torch.allclose(points_sparse.cpu(), pc0[indices_A_to_C])
            #     if "new_lidar_sparse" in f[key]:
            #         del f[key]["new_lidar_sparse"]ssss
            #     if "new_lidar_neighbor_sparse" in f[key]:
            #         del f[key]["new_lidar_neighbor_sparse"]
            #     if "all_to_new_lidar_sparse_idx" in f[key]:
            #         del f[key]["all_to_new_lidar_sparse_idx"]
            #     f[key].create_dataset("new_lidar_sparse", data = points_sparse.cpu().numpy().astype(np.float32))
            #     f[key].create_dataset("new_lidar_neighbor_sparse", data = neighbor.cpu().numpy().astype(np.float32))
            #     f[key].create_dataset("all_to_new_lidar_sparse_idx", data = indices_A_to_C.cpu().numpy().astype(np.float32))
            
            # del pc0_all, pc0_one
            # del voxel_info_dict_all, voxel_info_dict_all_sparse
            # del points_all, coordinates_all, points_sparse, coordinates_sparse, point_idxes_sparse
            # del neighbor, indices_A_to_B, indices_A_to_C

            # # 释放 GPU 缓存（减少显存占用）
            # torch.cuda.empty_cache()


        return res_dict







class HDF5Dataset_onlybox_multiframe(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_onlybox_multiframe, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('/data0/code/Flow4D_croco/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                index_ = index_ - 1
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            # print(f[key].keys()) 'ego_motion', 'flow', 'flow_category_indices', 'flow_is_valid', 'lidar', 'pose'
            pc0 = torch.tensor(f[key]['lidar'][:]) 
            # gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 

            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            # gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                # 'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                # 'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    # past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict
    
def collate_fn_pad_onlybox(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground= [], []
    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['pc0'][~(batch[i]['gm0'] | (batch[i]['flow_category_indices'] == 0))])
        pc1_after_mask_ground.append(batch[i]['pc1'][~(batch[i]['gm1'] | (batch[i]['flow_category_indices_pc1'] == 0))])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][~(batch[i]['gm0'] | (batch[i]['flow_category_indices'] == 0))] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][~(batch[i]['gm0'] | (batch[i]['flow_category_indices'] == 0))] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][~(batch[i]['gm0'] | (batch[i]['flow_category_indices'] == 0))] for i in range(len(batch))], batch_first=True)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][~(batch[i]['gm0'] | (batch[i]['flow_category_indices'] == 0))] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict

class HDF5Dataset_onlybox(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_onlybox, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('/data0/code/Flow4D_croco/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if (self.scene_id_bounds[scene_id]["max_index"]- 1) <= index_: 
                index_ = index_ - 2
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            pc0 = torch.tensor(f[key]['lidar'][:]) 
            gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 

            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices # 使用原始的单纯背景类别0（不进行映射）过滤
                flow_category_indices_pc1 = torch.tensor(f[next_timestamp]['flow_category_indices'][:]) #!add
                res_dict['flow_category_indices_pc1'] = flow_category_indices_pc1


                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict
# diff完成稀疏到稠密的生成
def collate_fn_pad_new_NK(batch):

    num_frames = 2
    while f'pc_m{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_all_after_mask_ground, pc1_all_after_mask_ground= [], []
    pc0_one_after_mask_ground, pc1_one_after_mask_ground= [], []
    pc0_a_to_c_list = []
    pc1_a_to_c_list = []
    time_list = []
    scene_id_list = []

    pc_m_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_a_to_c = batch[i]['pc0_neighbor_a_to_c'].to(torch.long)
        pc0_a_to_c_list.append(pc0_a_to_c)
        pc1_a_to_c = batch[i]['pc1_neighbor_a_to_c'].to(torch.long)
        pc1_a_to_c_list.append(pc1_a_to_c)

        pc0_all_after_mask_ground.append(batch[i]['pc0_neighbor'])
        pc1_all_after_mask_ground.append(batch[i]['pc1_neighbor'])
        pc0_one_after_mask_ground.append(batch[i]['pc0'][pc0_a_to_c])
        pc1_one_after_mask_ground.append(batch[i]['pc1'][pc1_a_to_c])
        time_list.append(batch[i]['timestamp'])
        scene_id_list.append(batch[i]['scene_id'])
        for j in range(1, num_frames - 1):
            pc_m_after_mask_ground[j-1].append(batch[i][f'pc_m{j}'][~batch[i][f'gm_m{j}']])
    

    pc0_all_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_all_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_all_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_all_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc0_one_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_one_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_one_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_one_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc_m_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pc_m, batch_first=True, padding_value=torch.nan) for pc_m in pc_m_after_mask_ground]


    res_dict =  {
        'pc0_all': pc0_all_after_mask_ground,
        'pc1_all': pc1_all_after_mask_ground,
        'pc0': pc0_one_after_mask_ground,
        'pc1': pc1_one_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))],
        'timestamps': time_list,
        'scene_ids': scene_id_list,
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pc_m{j}'] = pc_m_after_mask_ground[j-1]
        res_dict[f'pose_m{j}'] = [batch[i][f'pose_m{j}'] for i in range(len(batch))]

    if 'flow' in batch[0]:
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][pc0_a_to_c[i]] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][pc0_a_to_c[i]] for i in range(len(batch))], batch_first=True) #! 不需要class mask，因为是目标帧的输出
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][pc0_a_to_c[i]] for i in range(len(batch))], batch_first=True,  padding_value=255)
        flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_labeled'][pc0_a_to_c[i]] for i in range(len(batch))], batch_first=True)
        
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices
        res_dict['flow_category_labeled'] = flow_category_labeled

        neighbor_flow = torch.nn.utils.rnn.pad_sequence([batch[i]['neighbor_flow']for i in range(len(batch))], batch_first=True)
        neighbor_flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['neighbor_flow_is_valid'] for i in range(len(batch))], batch_first=True) #! 不需要class mask，因为是目标帧的输出
        neighbor_flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['neighbor_flow_category_indices'] for i in range(len(batch))], batch_first=True,  padding_value=255)
        neighbor_flow_category_labeled = torch.nn.utils.rnn.pad_sequence([batch[i]['neighbor_flow_category_labeled'] for i in range(len(batch))], batch_first=True)
        
        res_dict['neighbor_flow'] = neighbor_flow
        res_dict['neighbor_flow_is_valid'] = neighbor_flow_is_valid
        res_dict['neighbor_flow_category_indices'] = neighbor_flow_category_indices
        res_dict['neighbor_flow_category_labeled'] = neighbor_flow_category_labeled

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]

    return res_dict



#! multi_frame_idx 多帧加载监督N,K,3个点，配合collate_new_NK
class HDF5Dataset_new_NK(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_new_NK, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
        if self.mode == 'train':
            self.data_index = self.data_index[:10] #! debug
        # self.data_index = sorted(self.data_index) # sorted for debug
        # self.data_index = sorted(self.data_index) # sorted for debug

        with open('/data0/code/Flow4D_diff_less_to_more/conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if (self.scene_id_bounds[scene_id]["max_index"] - 1) <= index_: #! 因为最后一阵没有target_mask
                index_ = index_ - 2
            scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 

            pc0 = torch.tensor(f[key]['target_point'][:]) 
            gm0 = torch.tensor(f[key]['target_ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 

            pc0_neighbor = torch.tensor(f[key]['neighbor_point'][:])
            pc0_neighbor_a_to_c = torch.tensor(f[key]['neighbor_a_to_c'][:])

            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['target_point'][:]) 
            gm1 = torch.tensor(f[next_timestamp]['target_ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:]) 

            pc1_neighbor = torch.tensor(f[next_timestamp]['neighbor_point'][:])
            pc1_neighbor_a_to_c = torch.tensor(f[next_timestamp]['neighbor_a_to_c'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,

                "pc0": pc0,
                "gm0": gm0,
                "pose0": pose0,

                "pc0_neighbor": pc0_neighbor,
                "pc0_neighbor_a_to_c": pc0_neighbor_a_to_c,

                "pc1": pc1,
                "gm1": gm1,
                "pose1": pose1,

                "pc1_neighbor": pc1_neighbor,
                "pc1_neighbor_a_to_c": pc1_neighbor_a_to_c
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose


            if 'target_flow' in f[key]:
                target_flow = torch.tensor(f[key]['target_flow'][:])
                target_flow_valid = torch.tensor(f[key]['target_flow_valid'][:]) 
                target_flow_category = torch.tensor(f[key]['target_flow_category'][:]) 

                neighbor_flow = torch.tensor(f[key]['neighbor_flow'][:])
                neighbor_flow_vaild = torch.tensor(f[key]['neighbor_flow_vaild'][:]) 
                neighbor_flow_category = torch.tensor(f[key]['neighbor_flow_category'][:]) 

                res_dict['flow'] = target_flow
                res_dict['flow_is_valid'] = target_flow_valid
                res_dict['flow_category_indices'] = target_flow_category
                res_dict['neighbor_flow'] = neighbor_flow
                res_dict['neighbor_flow_is_valid'] = neighbor_flow_vaild
                res_dict['neighbor_flow_category_indices'] = neighbor_flow_category

                flow_category_labeled = map_label(f[key]['target_flow_category'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor 

                neighbor_flow_category_labeled = map_label(f[key]['neighbor_flow_category'][:], self.learning_map) 
                neighbor_flow_category_labeled_tensor = torch.tensor(neighbor_flow_category_labeled, dtype=torch.int32)
                res_dict['neighbor_flow_category_labeled'] = neighbor_flow_category_labeled_tensor 

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion


            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict

def split_frame(frame_list, num):
    ts_list = []
    for frame in frame_list:
        ts_list.append(frame['timestamp'].decode('utf-8'))

    current_len = len(ts_list)
    if current_len < num:
        needed = num - current_len
        ts_list += [ts_list[i % current_len] for i in range(needed)]

    return ts_list[:num]

def cal_pose_inv(pose1: np.ndarray):
    pose1_inv = np.eye(4, dtype=np.float64)
    R = pose1[:3, :3]
    t = pose1[:3, 3]
    pose1_inv[:3, :3] = R.T  # 旋转部分取转置
    pose1_inv[:3, 3] = -R.T @ t  # 正确的矩阵乘法计算平移逆

    return pose1_inv


def save_lidar(pt_np, path, color = [0, 1.0, 0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_np)  # 100个随机点
    pcd.paint_uniform_color(color)  # 设置颜色（绿色）
    o3d.io.write_point_cloud(path, pcd)


class HDF5Dataset_selectbox(Dataset):
    def __init__(self, directory, n_frames, eval = False):
        '''
        directory: the directory of the dataset
        eval: if True, use the eval index
        '''
        super(HDF5Dataset_selectbox, self).__init__()
        self.directory = directory
        self.mode = os.path.basename(self.directory)
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

        with open('./conf/labeling.yaml', 'r') as file:
            labeling_map = yaml.safe_load(file)

        self.learning_map = labeling_map['Argoverse_learning_map']

        self.n_frames = n_frames
        assert self.n_frames >= 2, "n_frames must be 2 or more."
        
        print('dataloader mode = {} num_frames = {}'.format(self.mode, self.n_frames))

        self.eval_index = False
        # ! eval_all image
        if eval:
            if not os.path.exists(os.path.join(self.directory, 'index_eval.pkl')):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval

            if self.mode == 'val':
                with open(os.path.join(self.directory, 'index_eval.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            elif self.mode == 'test':
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f: #jy
                    self.eval_data_index = pickle.load(f)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    
    def __getitem__(self, index_):
        #! eval all
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp]) 
        else:
            scene_id, timestamp = self.data_index[index_] 
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_: 
                index_ = index_ - 1
        scene_id, timestamp = self.data_index[index_] 

        key = str(timestamp)
        # with h5py.File(os.path.join('/data1/dataset/av2/box_visual/sensor/val/', f'{scene_id}.h5'), 'r') as f: 
        #     pc0_box = torch.tensor(f[key]['lidar'][:]) 
        #     gm0_box = torch.tensor(f[key]['ground_mask'][:]) 
        #     pose0_box = torch.tensor(f[key]['pose'][:]) 

        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f: 
            pc0 = torch.tensor(f[key]['lidar'][:]) 
            gm0 = torch.tensor(f[key]['ground_mask'][:]) 
            pose0 = torch.tensor(f[key]['pose'][:]) 
            #! add_select_box
            box_ids_dset0 = f[key]['box_ids']
            box_ids0 = [x.decode('utf-8') for x in box_ids_dset0[:]] #0时刻下的box_id
            box_poses0 = f[key]['box_poses']
            acc_points0 = []
            for i, box_id in enumerate(box_ids0): #对每一个box进行筛选
                box_group = f[box_id]
                select_best = box_group['sorted_frame_list']
                select_best_sorted = split_frame(select_best, 8)
                select_point_list = []
                for best_frame in select_best_sorted: #对每一个最佳帧进行筛选
                    point = box_group[best_frame]['point']
                    pose = box_group[best_frame]['pose']
                    pose_inv = cal_pose_inv(pose)
                    transformed_point = point @ pose_inv[:3, :3].T + pose_inv[:3, 3]
                    select_point_list.append(transformed_point)
                select_point_np = np.concatenate(select_point_list, axis=0)
                box_pose0 = box_poses0[i]
                select_point_transformed = select_point_np @ box_pose0[:3, :3].T + box_pose0[:3, 3]
                acc_points0.append(select_point_transformed)
            acc_points0_all = np.concatenate(acc_points0, axis=0)
            os.makedirs(f"./{scene_id}", exist_ok=True)
            save_lidar(acc_points0_all, f"./{scene_id}/{key}_acc_points0.ply")
            save_lidar(pc0, f"./{scene_id}/{key}_pc0.ply")
            all_points = np.concatenate([pc0, acc_points0_all], axis=0)
            save_lidar(all_points, f"./{scene_id}/{key}_all_points.ply")
            # assert torch.equal(pc0_box, pc0)
            # assert torch.equal(gm0_box, gm0)
            # assert torch.equal(pose0_box, pose0)

            if self.scene_id_bounds[scene_id]["max_index"] == index_:
                print("!!!!!!__getitem__(index_ + 1)") 
                return self.__getitem__(index_ + 1)
            else:
                next_timestamp = str(self.data_index[index_+1][1])

            pc1 = torch.tensor(f[next_timestamp]['lidar'][:])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:]) 
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])


            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0, #current
                'gm0': gm0, #current
                'pose0': pose0, #current
                'pc1': pc1, #nect
                'gm1': gm1, #next
                'pose1': pose1, #next
            }


            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pc_m{i+1}'] = past_pc
                    res_dict[f'gm_m{i+1}'] = past_gm
                    res_dict[f'pose_m{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:]) 
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:]) 
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices #原始的category属性
                mask = (flow_category_indices == 0) & (~gm0)
                no_box_background = pc0[mask]
                save_lidar(all_points, f"./{scene_id}/{key}_no_box_background.ply")
                flow_category_labeled = map_label(f[key]['flow_category_indices'][:], self.learning_map) 
                flow_category_labeled_tensor = torch.tensor(flow_category_labeled, dtype=torch.int32)
                res_dict['flow_category_labeled'] = flow_category_labeled_tensor #映射之后的label

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

            #! eval all
            # res_dict['eval_mask'] = (~(gm0 | (torch.tensor(f[key]['flow_category_indices'][:]) == 0)))
            if self.eval_index: 
                if self.mode == 'val':
                    eval_mask = torch.tensor(f[key]['eval_mask'][:])
                    res_dict['eval_mask'] = eval_mask 
                elif self.mode == 'test':
                    eval_mask = torch.ones(pc0.shape[0], 1, dtype=torch.bool) 
                    res_dict['eval_mask'] = eval_mask
                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Only 'val' and 'test' are supported.")

        return res_dict


import numpy as np
from scipy.spatial.transform import Rotation as R

def add_se3_noise(T_gt, trans_std=0.1, angle_std=10.0):
    """
    为4×4变换矩阵添加平移和旋转噪声
    
    参数:
        T_gt: 原始4×4变换矩阵 (numpy数组)
        trans_std: 平移噪声标准差 (单位：米)
        angle_std: 旋转噪声标准差 (单位：度)
    
    返回:
        T_noisy: 加噪后的4×4矩阵
    """
    # 分解矩阵为旋转和平移
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    
    # --- 平移噪声 ---
    delta_t = np.random.normal(0, trans_std, 3)
    t_noisy = t_gt + delta_t
    
    # --- 旋转噪声 ---
    # 生成随机轴-角扰动 (角度符合高斯分布)
    axis = np.random.randn(3)          # 随机轴（归一化前）
    axis /= np.linalg.norm(axis)       # 单位化
    angle_rad = np.deg2rad(np.random.normal(0, angle_std))  # 角度转弧度
    
    # 生成扰动旋转矩阵
    delta_R = R.from_rotvec(axis * angle_rad).as_matrix()
    R_noisy = R_gt @ delta_R  # 矩阵乘法（右乘扰动）
    
    # 重建噪声矩阵
    T_noisy = np.eye(4)
    T_noisy[:3, :3] = R_noisy
    T_noisy[:3, 3] = t_noisy
    return T_noisy



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader test")
    parser.add_argument('--data_mode', '-m', type=str, default='val', metavar='N', help='Dataset mode.')
    parser.add_argument('--data_path', '-d', type=str, default='/data1/dataset/av2/multi_frame_idx/sensor/', metavar='N', help='preprocess data path.')
    options = parser.parse_args()

    # testing eval mode
    dataset = HDF5Dataset_multi_frame_idx_changedataloader(options.data_path+"/"+options.data_mode, n_frames=2, eval=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for data in tqdm(dataloader, ncols=80, desc="eval mode"):
        res_dict = data
