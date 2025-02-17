
"""
# Created: 2023-07-18 15:08
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""

import torch.nn as nn
import dztimer, torch

from .basic.embedder_model_flow4D import DynamicEmbedder_4D_NK_onlyrestore
from .basic import cal_pose0to1

from .basic.network_4D import Network_4D, Seperate_to_3D, Point_head
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out
    

class Flow4D(nn.Module):
    def __init__(self, voxel_size = [0.2, 0.2, 0.2],
                 point_cloud_range = [-51.2, -51.2, -2.2, 51.2, 51.2, 4.2],
                 grid_feature_size = [512, 512, 32],
                 num_frames = 5):
        super().__init__()

        point_output_ch = 16
        voxel_output_ch = 16

        self.num_frames = num_frames
        print('voxel_size = {}, pseudo_dims = {}, input_num_frames = {}'.format(voxel_size, grid_feature_size, self.num_frames))

        # self.embedder_4D = DynamicEmbedder_4D_ori(voxel_size=voxel_size,
        #                                 pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
        #                                 point_cloud_range=point_cloud_range,
        #                                 feat_channels=point_output_ch)
        
        # self.network_4D = Network_4D(in_channel=point_output_ch, out_channel=voxel_output_ch)

        # self.seperate_feat = Seperate_to_3D(num_frames)

        # self.pointhead_3D = Point_head(voxel_feat_dim=voxel_output_ch, point_feat_dim=point_output_ch)
        
        self.timer = dztimer.Timing()
        self.timer.start("Total")

        self.embedder_4D_offset_add_noise = DynamicEmbedder_4D_NK_onlyrestore(voxel_size=voxel_size,
                                pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
                                point_cloud_range=point_cloud_range,
                                feat_channels=point_output_ch)
        # self.embedder_4D_restore = DynamicEmbedder_4D_restore(voxel_size=voxel_size,
        #                         pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
        #                         point_cloud_range=point_cloud_range,
        #                         feat_channels=point_output_ch)
        # self.pc0_feature = DynamicEmbedder_4D_pc0(voxel_size=voxel_size,
        #                         pseudo_image_dims=[grid_feature_size[0], grid_feature_size[1], grid_feature_size[2], num_frames], 
        #                         point_cloud_range=point_cloud_range,
        #                         feat_channels=point_output_ch)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, batch, training_flag):
        #t_deflow_start = time.time()
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """

        batch_sizes = len(batch["pose0"])
        # pc1s = batch["pc1"]


        pcs_dict = {
            'pc0s_ori': batch["pc0"],
            # 'pc1s': pc1s,
            'timestamps': batch["timestamps"],
            'scene_ids': batch["scene_ids"],
        }

        if training_flag:#!
            # pc0_all = torch.stack(change_after['pc0_all'], dim=0) 
            # pcs_dict['pc0_all_change'] = pc0_all #! transform pc0_all 坐标到第一帧坐标系下
            pcs_dict['pc0_all_ori'] = batch["pc0_all"].reshape(batch_sizes,-1,3)
            # pcs_dict['pc1_all'] = batch["pc1_all"].reshape(batch_sizes,-1,3)


        for i in range(1, self.num_frames - 1):
            pcs_dict[f'pc_m{i}s'] = pc_m_frames[i-1]


        self.timer[1].start("4D_voxelization")
        if training_flag:
            restore_point_dict, restore_loss= self.embedder_4D_offset_add_noise(pcs_dict, training_flag)
        else:
            restore_point_dict = self.embedder_4D_offset_add_noise(pcs_dict, training_flag)

        # dict_4d = self.embedder_4D(pcs_dict, True)
        restore_point_dict_batch = {}
        device = batch["pc0"].device
        
        padding_value = torch.tensor(float('nan'), device=device)
        for key, value in restore_point_dict.items():
            restore_point_dict_batch[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=padding_value) #torch.nan)


        restore_pc0_all = restore_point_dict_batch['pc0_all_ori']
        # restore_pc1_all = restore_point_dict_batch['pc1_all']
        # restore_all_dict = {'pc0s_restore': restore_point_dict_batch['pc0_all'], 'pc1s_restore': restore_point_dict_batch['pc1_all']}
        restore_all_dict = {'pc0s_restore': restore_pc0_all, \
                            'pc0s_gt': batch["pc0_all"].reshape(batch_sizes,-1,3),
                            }
        
        return restore_all_dict
        #! 只有数据恢复第一阶段
        # dict_4d_restore = self.embedder_4D_restore(restore_all_dict)  #! freeze 
        # pc01_tesnor_4d = dict_4d_restore['4d_tensor']

        # # dict_4d = self.pc0_feature(pc0s.clone()) 
        # pc0_3dvoxel_infos_lst =dict_4d_restore['pc0_3dvoxel_infos_lst']
        # pc0_point_feats_lst =dict_4d_restore['pc0_point_feats_lst']
        # pc0_num_voxels = dict_4d_restore['pc0_mum_voxels']

        # # pc01_tesnor_4d = dict_4d['4d_tensor']
        # # pc0_3dvoxel_infos_lst =dict_4d['pc0_3dvoxel_infos_lst']
        # # pc0_point_feats_lst =dict_4d['pc0_point_feats_lst']
        # # pc0_num_voxels = dict_4d['pc0_mum_voxels']
        # self.timer[1].stop()

        # self.timer[2].start("4D_backbone")
        # pc_all_output_4d = self.network_4D(pc01_tesnor_4d) #all = past, current, next 다 합친것
        # self.timer[2].stop()

        # self.timer[3].start("4D pc01 to 3D pc0")
        # pc0_last = self.seperate_feat(pc_all_output_4d)
        # # assert pc0_last.features.shape[0] == pc0_num_voxels, 'voxel number mismatch'
        # self.timer[3].stop()

        # self.timer[4].start("3D_sparsetensor_to_point and head")
        # flows = self.pointhead_3D(pc0_last, pc0_3dvoxel_infos_lst, pc0_point_feats_lst)
        # self.timer[4].stop()

        # pc0_points_lst = [e["points"] for e in pc0_3dvoxel_infos_lst] 
        # pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_3dvoxel_infos_lst] 

        # model_res = {
        #     "flow": flows, 
        #     'pose_flow': pose_flows_pc0, #! for pc0

        #     "pc0_valid_point_idxes": pc0_valid_point_idxes, 
        #     "pc0_points_lst": pc0_points_lst, 
            
        # }
        # if training_flag:
        #     model_res['restore_loss'] = restore_loss
