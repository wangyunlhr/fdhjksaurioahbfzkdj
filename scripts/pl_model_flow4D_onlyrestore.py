"""

# Created: 2023-11-05 10:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Model Wrapper for Pytorch Lightning

"""

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf,open_dict

import os, sys, time, h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from scripts.utils.mics import import_func, weights_init, zip_res
from scripts.utils.av2_eval import write_output_file, write_output_file_2023, write_output_file_v2, compute_class_loss, compute_point_epe
from scripts.network.models.basic import cal_pose0to1
from scripts.network.official_metric import PointMetrics, evaluate_leaderboard, evaluate_leaderboard_v2
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()



torch.set_float32_matmul_precision('medium')
class ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()

        # set grid size
        if ('voxel_size' in cfg.model.target) and ('point_cloud_range' in cfg.model.target) and not eval and 'point_cloud_range' in cfg:
            OmegaConf.set_struct(cfg.model.target, True)
            with open_dict(cfg.model.target):
                cfg.model.target['grid_feature_size'] = \
                    [abs(int((cfg.point_cloud_range[0] - cfg.point_cloud_range[3]) / cfg.voxel_size[0])),
                    abs(int((cfg.point_cloud_range[1] - cfg.point_cloud_range[4]) / cfg.voxel_size[1])),
                    abs(int((cfg.point_cloud_range[2] - cfg.point_cloud_range[5]) / cfg.voxel_size[2]))]
        else:
            with open_dict(cfg.model.target):
                cfg.model.target['grid_feature_size'] = \
                    [abs(int((cfg.model.target.point_cloud_range[0] - cfg.model.target.point_cloud_range[3]) / cfg.model.target.voxel_size[0])),
                    abs(int((cfg.model.target.point_cloud_range[1] - cfg.model.target.point_cloud_range[4]) / cfg.model.target.voxel_size[1])),
                    abs(int((cfg.model.target.point_cloud_range[2] - cfg.model.target.point_cloud_range[5]) / cfg.model.target.voxel_size[2]))]
        
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)
        
        self.loss_fn = import_func("scripts.network.loss_func."+cfg.loss_fn) if 'loss_fn' in cfg else None
        self.loss_fn_restore = import_func("scripts.network.loss_func.restoreLoss")
        print('flow_loss func = {}'.format(cfg.loss_fn))
        self.batch_size = int(cfg.batch_size) if 'batch_size' in cfg else 1
        self.lr = cfg.lr if 'lr' in cfg else None
        self.epochs = cfg.epochs if 'epochs' in cfg else None

        if not hasattr(cfg, 'submit_version') or cfg.submit_version is None:
            self.submit_version = '2024'
        else:
            cfg.submit_version = str(cfg.submit_version)
            if cfg.submit_version == '2023':
                self.submit_version = '2023' #2023 challenge test server submit
            elif cfg.submit_version == '2024':
                self.submit_version = '2024' #2024 challenge test server submit
            else:
                raise ValueError(f"Invalid version: {cfg.submit_version}. Submit Version must be '2023' or '2024'.")
            
        print('submit_version = {}'.format(self.submit_version))


        self.metrics = PointMetrics()

        if 'checkpoint' in cfg:
            self.load_checkpoint_path = cfg.checkpoint

        if 'av2_mode' in cfg:
            self.av2_mode = cfg.av2_mode
            self.save_res = cfg.save_res
            if self.save_res:
                self.save_res_path = Path('/data2/deflow/results') / cfg.output
                os.makedirs(self.save_res_path, exist_ok=True)
                print(f"We are in {cfg.av2_mode}, results will be saved in: {self.save_res_path}")
        else:
            self.av2_mode = None
            if 'pretrained_weights' in cfg:
                if cfg.pretrained_weights is not None:
                    # self.model.load_from_checkpoint(cfg.pretrained_weights) #! no strict load全部参数
                    # #! only load encoder weights
                    pretrained_state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")  # 加载权重文件

                    model_state_dict = self.model.state_dict()
                    # # 加载权重时只匹配部分参数
                    filtered_weights = {k[len("model."):].replace("embedder_4D", "embedder_4D_restore", 1): v for k, v in pretrained_state_dict['state_dict'].items() if k.startswith('model.embedder_4D')}
                    model_state_dict.update(filtered_weights)
                    self.model.load_state_dict(model_state_dict)
                    # 冻结加载的部分权重
                    for name, param in self.model.named_parameters():
                        # print(name)
                        if name.startswith('embedder_4D_restore'):  # 只冻结以 'embedder_4D' 为前缀的参数
                            param.requires_grad = False  # 冻结该部分权重



        if 'dataset_path' in cfg:
            self.dataset_path = cfg.dataset_path
        if 'res_name' in cfg:
            self.vis_name = cfg.res_name
        self.save_hyperparameters()



    def fast_chamfer(self, A, B, truncate_dist=None):
        """
        计算 Chamfer 距离，适用于 (N, K, 3) 结构，并支持距离截断。

        A: (N, K, 3) - 第一个点云
        B: (N, K, 3) - 第二个点云
        truncate_dist: float, 如果不为 None,则截断距离超过该值的点对
        返回值: (N,) - 每个 voxel 内的 Chamfer 距离
        """
        # 计算点到点的欧式距离
        dist_mat = torch.cdist(A, B, p=2)  # (N, K, K)，每个 voxel 内 K 个点的距离矩阵

        if truncate_dist is not None:
            # 设定截断：超过 `truncate_dist` 的距离设为无效（inf）
            dist_mat = torch.where(dist_mat > truncate_dist, torch.tensor(float('inf'), device=dist_mat.device), dist_mat)

        # 每个点找到最近的匹配点
        min_dist_A_to_B, _ = torch.min(dist_mat, dim=2)  # (N, K)
        min_dist_B_to_A, _ = torch.min(dist_mat, dim=1)  # (N, K)

        if truncate_dist is not None:
            # 只计算有效的（非 `inf`）距离
            valid_A = min_dist_A_to_B != float('inf')
            valid_B = min_dist_B_to_A != float('inf')

            chamfer_A = torch.where(valid_A, min_dist_A_to_B, torch.tensor(0.0, device=A.device)).sum(dim=1) / valid_A.sum(dim=1).clamp(min=1)
            chamfer_B = torch.where(valid_B, min_dist_B_to_A, torch.tensor(0.0, device=A.device)).sum(dim=1) / valid_B.sum(dim=1).clamp(min=1)

            chamfer_dist = chamfer_A + chamfer_B  # (N,)
        else:
            # 直接计算 Chamfer 距离（无截断）
            chamfer_dist = min_dist_A_to_B.mean(dim=1) + min_dist_B_to_A.mean(dim=1)  # (N,)

        return chamfer_dist



    def training_step(self, batch, batch_idx):
        self.model.timer[4].start("One Scan in model")
        res_dict = self.model(batch, True)
        self.model.timer[4].stop()

        self.model.timer[5].start("Loss")

        # compute loss
        total_loss = 0.0

        batch_sizes = len(batch["pose0"])
        # restore_all_dict = {'pc0s_restore': restore_pc0_all, 'pc1s_restore': restore_pc1_all, \
        #             'pc0s_gt': batch["pc0_all"].reshape(batch_sizes,-1,3),
        #             'pc1s_gt': batch["pc1_all"].reshape(batch_sizes,-1,3),
        #             }
        pc0s_restore = res_dict['pc0s_restore']
        pc0s_gt = res_dict['pc0s_gt']
        
        # gt_flow = batch['flow'] #gt_flow = ego+motion

        # pose_flows = res_dict['pose_flow'] #pose_flow = ego-motion's flow
        # pc0_valid_idx = res_dict['pc0_valid_point_idxes'] # since padding
        # est_flow = res_dict['flow'] #network's output, motion flow 
        # restore_loss_list = res_dict['restore_loss']

        
        for batch_id in range(batch_sizes):
            restore_point = pc0s_restore[batch_id]
            restore_not_nan_mask = ~torch.isnan(restore_point).any(dim=1)
            restore_point_valid = restore_point[restore_not_nan_mask]
            gt_point = pc0s_gt[batch_id]
            gt_not_nan_mask = ~torch.isnan(gt_point).any(dim=1)
            gt_point_valid = gt_point[gt_not_nan_mask]
            assert torch.equal(restore_not_nan_mask, gt_not_nan_mask)
            gt_class = batch['flow_category_indices'][batch_id]
            class_not_nan_mask = ~ (gt_class == 255)
            gt_class_valid = gt_class[class_not_nan_mask].unsqueeze(1).repeat(1, 5).reshape(-1)
            assert gt_class_valid.shape[0] == restore_point_valid.shape[0]

            loss = compute_class_loss(pred_point = restore_point_valid, gt_point = gt_point_valid, category_indices = gt_class_valid)
            #! change loss
            chamferdis = self.fast_chamfer(restore_point_valid.reshape(-1, 5, 3), gt_point_valid.reshape(-1, 5, 3), 4.0)

            # loss_restore1 = self.loss_fn_restore(restore_loss_list[batch_id])
            # loss_restore2 = self.loss_fn_restore(restore_loss_list[batch_id + batch_sizes])
            # total_flow_loss += loss.item()

            # total_restore_loss += loss_restore1 + loss_restore2
            total_loss += loss
        
        self.log("trainer/loss", (total_loss.item())/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        # self.log("trainer/restore_loss", (total_restore_loss.item())/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        # total_loss += total_restore_loss
        # self.log("trainer/all_loss", (total_loss.item())/batch_sizes, sync_dist=True, batch_size=self.batch_size)
        # print("total loss", total_loss)
        # print("restore loss", restore_loss)
        # total_loss += restore_loss
        self.model.timer[5].stop()
        # NOTE (Qingwen): if you want to view the detail breakdown of time cost
        # self.model.timer.print(random_colors=False, bold=False)
        return total_loss

    def train_validation_step_(self, batch, res_dict): 
        # means there are ground truth flow so we can evaluate the EPE-3 Way metric
        # if batch['flow'][0].shape[0] > 0:
            # pose_flows = res_dict['pose_flow']
        batch_sizes = len(batch["pose0"])
        # restore_all_dict = {'pc0s_restore': restore_pc0_all, 'pc1s_restore': restore_pc1_all, \
        #             'pc0s_gt': batch["pc0_all"].reshape(batch_sizes,-1,3),
        #             'pc1s_gt': batch["pc1_all"].reshape(batch_sizes,-1,3),
        #             }
        pc0s_restore = res_dict['pc0s_restore']
        pc0s_gt = res_dict['pc0s_gt']
        for batch_id, gt_flow in enumerate(batch["flow"]):
            restore_point = pc0s_restore[batch_id]
            restore_not_nan_mask = ~torch.isnan(restore_point).any(dim=1)
            restore_point_valid = restore_point[restore_not_nan_mask]
            gt_point = pc0s_gt[batch_id]
            gt_not_nan_mask = ~torch.isnan(gt_point).any(dim=1)
            gt_point_valid = gt_point[gt_not_nan_mask]
            assert torch.equal(restore_not_nan_mask, gt_not_nan_mask)
            gt_class = batch['flow_category_indices'][batch_id]
            class_not_nan_mask = ~ (gt_class == 255)
            gt_class_valid = gt_class[class_not_nan_mask].unsqueeze(1).repeat(1, 5).reshape(-1)
            assert gt_class_valid.shape[0] == restore_point_valid.shape[0]


            point_dict = compute_point_epe(
                restore_point_valid.detach().cpu().numpy().astype(float),
                gt_point_valid.detach().cpu().numpy().astype(float),
                gt_class_valid.detach().cpu().numpy().astype(np.uint8),
            )
            self.metrics.step(point_dict)
        else:
            pass
        
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)

        if self.av2_mode == 'test':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"Test results saved in: {self.save_res_path}, Please run submit to zip the results and upload to online leaderboard.")
            return
        
        if self.av2_mode == 'val':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
            print(f"More details parameters and training status are in checkpoints")        

        self.metrics.normalize()

        # wandb log things:
        for key in self.metrics.point_bag:
            self.log(f"val/{key}", self.metrics.point_bag[key], sync_dist=True)
        
        self.metrics.print()
        self.metrics = PointMetrics()
        
    def eval_only_step_(self, batch, res_dict):
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

        eval_mask = batch['eval_mask'].squeeze()
        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        if 'pc0_valid_point_idxes' in res_dict:
            if self.av2_mode == 'val':
                valid_from_pc2res = res_dict['pc0_valid_point_idxes'] 
                # flow in the original pc0 coordinate
                pred_flow = pose_flow[~batch['gm0']].clone() 
                pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']
                final_flow = pose_flow.clone() 
                final_flow[~batch['gm0']] = pred_flow 
            elif self.av2_mode == 'test':
                valid_from_pc2res = res_dict['pc0_valid_point_idxes']
                pred_flow = torch.zeros_like(pose_flow[~batch['gm0']])
                final_flow = torch.zeros_like(pose_flow)

                pred_flow[valid_from_pc2res] = res_dict['flow']
                final_flow[~batch['gm0']] = pred_flow


        if self.av2_mode == 'val': 
            gt_flow = batch["flow"] 
            v1_dict = evaluate_leaderboard(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                       gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], \
                                       batch['flow_category_indices'][eval_mask])
            v2_dict = evaluate_leaderboard_v2(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                    gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
            
            self.metrics.step(v1_dict, v2_dict)


        
        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        if self.save_res or self.av2_mode == 'test': # test must save data to submit in the online leaderboard.    
            save_pred_flow = final_flow[eval_mask, :3].cpu().detach().numpy()
            sweep_uuid = (batch['scene_id'], batch['timestamp'])

            if self.submit_version == '2024': #2024 challenge 
                write_output_file(save_pred_flow, sweep_uuid, self.save_res_path)
            elif self.submit_version == '2023': #2023 challenge 
                rigid_flow = pose_flow[eval_mask, :3].cpu().detach().numpy()
                is_dynamic = np.linalg.norm(save_pred_flow - rigid_flow, axis=1, ord=2) >= 0.05
                write_output_file_2023(save_pred_flow, is_dynamic, sweep_uuid, self.save_res_path)
            else:
                raise ValueError(f"Invalid version: {self.submit_version}. Submit Version must be '2023' or '2024'.")

    def validation_step(self, batch, batch_idx):
        if self.av2_mode == 'val' or self.av2_mode == 'test':
            batch['origin_pc0'] = batch['pc0'].clone()
            batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
            batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)

            num_frames = 2
            while f'pc_m{num_frames - 1}' in batch:
                num_frames += 1

            for j in range(1, num_frames - 1):
                batch[f'pc_m{j}'] = batch[f'pc_m{j}'][~batch[f'gm_m{j}']].unsqueeze(0)

            self.model.timer[12].start("One Scan")
            res_dict = self.model(batch, False)
            self.model.timer[12].stop()
            self.eval_only_step_(batch, res_dict)
        else:
            res_dict = self.model(batch, False)
            self.train_validation_step_(batch, res_dict)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self):
        self.time_start_train_epoch = time.time()

    def on_train_epoch_end(self):
        self.log("pre_epoch_cost (mins)", (time.time()-self.time_start_train_epoch)/60.0, on_step=False, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        # NOTE (Qingwen): again, val and test we only allow batch_size = 1
        batch['origin_pc0'] = batch['pc0'].clone()
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        res_dict = self.model(batch)
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if len(res_dict[key])>0}

        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        if 'pc0_valid_point_idxes' in res_dict:
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']

            # flow in the original pc0 coordinate
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']

            final_flow = pose_flow.clone()
            final_flow[~batch['gm0']] = pred_flow

        # write final_flow into the dataset.
        key = str(batch['timestamp'])
        scene_id = batch['scene_id']
        with h5py.File(os.path.join(self.dataset_path, f'{scene_id}.h5'), 'r+') as f:
            if self.vis_name in f[key]:
                del f[key][self.vis_name]
            f[key].create_dataset(self.vis_name, data=final_flow.cpu().detach().numpy().astype(np.float32))

    def on_test_epoch_end(self):
        print(f"\n\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.load_checkpoint_path}")
        print(f"We already write the estimate flow: {self.vis_name} into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
        print(f"python tests/scene_flow.py --flow_mode '{self.vis_name}' --data_dir {self.dataset_path}")
        print(f"Enjoy! ^v^ ------ \n")