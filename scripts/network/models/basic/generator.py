import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
# from utils import misc
# from utils.logger import *
# from SoftPool import soft_pool2d, SoftPool2d
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion import cosine_beta_schedule, default, SinusoidalPosEmb, extract_new
from network_4D import SpatioTemporal_Decomposition_Block, Seperate_to_3D
import spconv.pytorch as spconv
import random

def conv3x3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=stride,
                             padding=(1,1,1), bias=False, indice_key=indice_key)

def conv1x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1,1,1), stride=stride,
                             padding=0, bias=False, indice_key=indice_key)

def conv1x1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,3), stride=stride,
                             padding=(0,0,0,1), bias=False, indice_key=indice_key)

def conv3x3x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(3,3,3,1), stride=stride,
                             padding=(1,1,1,0), bias=False, indice_key=indice_key)

def conv1x1x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,1), stride=stride,
                             padding=0, bias=False, indice_key=indice_key)

class Seperate_to_pc1(nn.Module):
    def __init__(self,):
        super(Seperate_to_pc1, self).__init__()
        # self.num_frames = num_frames
        #self.return_pc1 = return_pc1

    def forward(self, sparse_4D_tensor):

        indices_4d = sparse_4D_tensor.indices
        features_4d = sparse_4D_tensor.features
        
        pc1_time_value = 1

        mask_pc1 = (indices_4d[:, -1] == pc1_time_value)
        
        pc1_indices = indices_4d[mask_pc1][:, :-1] 
        pc1_features = features_4d[mask_pc1]

        pc1_sparse_3D = sparse_4D_tensor.replace_feature(pc1_features)
        pc1_sparse_3D.spatial_shape = sparse_4D_tensor.spatial_shape[:-1]
        pc1_sparse_3D.indices = pc1_indices

        return pc1_sparse_3D
    
class Spatio_Block(nn.Module):
    def __init__(self, in_filters, mid_filters, out_filters, indice_key=None):
        super(Spatio_Block, self).__init__()


        self.act = nn.LeakyReLU()

        self.spatial_conv_1 = conv3x3x3(in_filters, mid_filters, indice_key=indice_key + "spbef")
        self.bn_s_1 = nn.BatchNorm1d(mid_filters)

        # self.temporal_conv_1 = conv1x1x1x3(in_filters, mid_filters)
        # self.bn_t_1 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_1 = conv1x1x1(mid_filters+in_filters, mid_filters, indice_key=indice_key + "sp1D")
        self.bn_fusion_1 = nn.BatchNorm1d(mid_filters)


        self.spatial_conv_2 = conv3x3x3(mid_filters, mid_filters, indice_key=indice_key + "spbef")
        self.bn_s_2 = nn.BatchNorm1d(mid_filters)

        # self.temporal_conv_2 = conv1x1x1x3(mid_filters, mid_filters)
        # self.bn_t_2 = nn.BatchNorm1d(mid_filters)

        self.fusion_conv_2 = conv1x1x1(mid_filters*2, out_filters, indice_key=indice_key + "sp1D")
        self.bn_fusion_2 = nn.BatchNorm1d(out_filters)

        # if self.pooling:
        #     if z_pooling == True:
        #         self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2,2,2,1), stride=(2,2,2,1), indice_key=down_key, bias=False)
        #     else:
        #         self.pool = spconv.SparseConv4d(out_filters, out_filters, kernel_size=(2,2,1,1), stride=(2,2,1,1), indice_key=down_key, bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        #ST block
        S_feat_1 = self.spatial_conv_1(x)
        S_feat_1 = S_feat_1.replace_feature(self.bn_s_1(S_feat_1.features))
        S_feat_1 = S_feat_1.replace_feature(self.act(S_feat_1.features))

        # T_feat_1 = self.temporal_conv_1(x)
        # T_feat_1 = T_feat_1.replace_feature(self.bn_t_1(T_feat_1.features))
        # T_feat_1 = T_feat_1.replace_feature(self.act(T_feat_1.features))

        ST_feat_1 = x.replace_feature(torch.cat([S_feat_1.features, x.features], 1)) #residual까지 concate

        ST_feat_1 = self.fusion_conv_1(ST_feat_1)
        ST_feat_1 = ST_feat_1.replace_feature(self.bn_fusion_1(ST_feat_1.features))
        ST_feat_1 = ST_feat_1.replace_feature(self.act(ST_feat_1.features))

        #TS block
        S_feat_2 = self.spatial_conv_2(ST_feat_1)
        S_feat_2 = S_feat_2.replace_feature(self.bn_s_2(S_feat_2.features))
        S_feat_2 = S_feat_2.replace_feature(self.act(S_feat_2.features))

        # T_feat_2 = self.temporal_conv_2(ST_feat_1)
        # T_feat_2 = T_feat_2.replace_feature(self.bn_t_2(T_feat_2.features))
        # T_feat_2 = T_feat_2.replace_feature(self.act(T_feat_2.features))

        ST_feat_2 = x.replace_feature(torch.cat([S_feat_2.features, ST_feat_1.features], 1)) #residual까지 concate
        
        ST_feat_2 = self.fusion_conv_2(ST_feat_2)
        ST_feat_2 = ST_feat_2.replace_feature(self.bn_fusion_2(ST_feat_2.features))
        ST_feat_2 = ST_feat_2.replace_feature(self.act(ST_feat_2.features))

        # if self.pooling: 
        #     pooled = self.pool(ST_feat_2)
        #     return pooled, ST_feat_2
        # else:
        return ST_feat_2







class VarianceSchedule(nn.Module):

    def __init__(self):
        super().__init__()
        
        # self.config = config
        # self.num_steps = self.config.generator_config.time_schedule.num_steps
        # self.beta_start = self.config.generator_config.time_schedule.beta_start
        # self.beta_end = self.config.generator_config.time_schedule.beta_end
        # self.mode = self.config.generator_config.time_schedule.mode
        
        # if self.mode == 'linear':
        #     betas = torch.linspace(self.beta_start, self.beta_end, steps=self.num_steps)
            
        # betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        # alphas = 1 - betas
        
        # alphas_cumprod = torch.cumprod(alphas, axis=0)

        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas', alphas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)

    # original sampling strategy
    # def uniform_sampling(self, batch_size):
    #     ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
    #     return ts.tolist()

    # Recurrent Uniform Sampling Strategy
        # !new code
        #! build diffusion
        timesteps = 1000
        sampling_timesteps = 1  #!sampling_timesteps
        self.timesteps = timesteps
        # define beta schedule
        betas = cosine_beta_schedule(timesteps=timesteps).float()
        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0
        # time embeddings
        time_dim = 4
        dim = 16
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # define alphas betas逐渐增大，噪声增多
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) #表示前面填充一位，后面填充零位
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.ddim_sampling_eta = 0.01
        self.scale = 1.0
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start, t, noise=None): #x_start gt数据
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract_new(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_new(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_noise_from_start(self, x_t, t, x0): #noise, t, pred residual flow
        return (
                (extract_new(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract_new(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise): #noise, t, pred residual flow
        return (
                extract_new(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - noise *
                extract_new(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def recurrent_uniform_sampling(self, batch_size, interval_nums):
        interval_size = self.timesteps / interval_nums
        sampled_intervals = []
        for i in range(interval_nums):
            start = int(i * interval_size)
            end = int((i + 1) * interval_size)
            sampled_interval = np.random.choice(np.arange(start, end), batch_size)
            sampled_intervals.append(sampled_interval)
        ts = np.vstack(sampled_intervals)
        ts = torch.tensor(ts)
        ts = torch.stack([ts[:, i][torch.randperm(interval_nums)] for i in range(batch_size)], dim=1)

        return ts
    def reconstruction(self, time, time_next, x0, pred_noise):
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        noise = (self.scale * torch.randn_like(x0)).float()
        x0_noise = x0 * alpha_next.sqrt() + \
            c * pred_noise + \
            sigma * noise
        return x0_noise
        


# Condition Aggregation Network
class CANet(nn.Module): 
    def __init__(self, encoder_dims, cond_dims):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.cond_dims = cond_dims

        self.mlp1 = nn.Sequential(
            nn.Conv2d(self.encoder_dims, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, self.cond_dims, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, patch_fea):
        '''
            patch_feature : B G 384
            -----------------
            point_condition : B 384
        '''
        
        patch_fea = patch_fea.transpose(1, 2)     # B 384 G
        patch_fea = patch_fea.unsqueeze(-1)       # B 384 G 1
        patch_fea = self.mlp1(patch_fea)          # B 512 G 1
        # soft_pool2d
        global_fea = soft_pool2d(patch_fea, kernel_size=[patch_fea.size(2), 1])  # B 512 1 1
        global_fea = global_fea.expand(-1, -1, patch_fea.size(2), -1)            # B 512 G 1
        combined_fea = torch.cat([patch_fea, global_fea], dim=1)                 # B 1024 G 1
        combined_fea = self.mlp2(combined_fea)                                       # B F G 1
        condition_fea = soft_pool2d(combined_fea, kernel_size=[combined_fea.size(2), 1])  # B F 1 1
        condition_fea = condition_fea.squeeze(-1).squeeze(-1)                          #  B F
        return condition_fea

# Point Condition Network 
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

# Point Denoising Network

class DenoisingNet(nn.Module):

    def __init__(self, point_dim, cond_dims, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        # self.layers = nn.ModuleList([
        #     PCNet(3, 128, cond_dims+3),
        #     PCNet(128, 256, cond_dims+3),
        #     PCNet(256, 512, cond_dims+3),
        #     PCNet(512, 256, cond_dims+3),
        #     PCNet(256, 128, cond_dims+3),
        #     PCNet(128, 3, cond_dims+3)
        # ])
        self.pred_noise = Spatio_Block(in_filters=16+16+4+16, mid_filters=32, out_filters=16, indice_key="cond_spatial")
        self.cond_spatial = nn.ModuleList([
                            SpatioTemporal_Decomposition_Block(in_filters=16, mid_filters=16, out_filters=16, indice_key="pn1"),
                            SpatioTemporal_Decomposition_Block(in_filters=16, mid_filters=16, out_filters=16, indice_key="pn2"),])
        self.seperate_pc1 = Seperate_to_pc1()
        self.seperate_pc0 = Seperate_to_3D(num_frames = 2)
    # def forward(self, coords, beta, cond):
    #     """
    #     Args:
    #         coords:   Noise point clouds at timestep t, (B, N, 3).
    #         beta:     Time. (B, ).
    #         cond:     Condition. (B, F).
    #     """

    #     batch_size = coords.size(0)
    #     beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
    #     cond = cond.view(batch_size, 1, -1)         # (B, 1, F)

    #     time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
    #     cond_emb = torch.cat([time_emb, cond], dim=-1)    # (B, 1, F+3)
        
    #     out = coords
    #     for i, layer in enumerate(self.layers):
    #         out = layer(fea=out, cond=cond_emb)
    #         if i < len(self.layers) - 1:
    #             out = self.act(out)

    #     if self.residual:
    #         return coords + out
    #     else:
    #         return out
    def forward(self, voxel_feature_pc1_noise, voxel_coor_pc1, beta, cond, batch_sizes, one_name):
        """
        Args:
            coords:   Noise point clouds at timestep t, (B, N, 3).
            beta:     Time. (B, time_dim=4).
            cond:     Condition. (N, C=16).  
        """
        spatial_shape = cond.spatial_shape
        cond_time = []
        cond_ori = cond
        # cond_process = self.cond_spatial(cond)
        for i, layer in enumerate(self.cond_spatial): #pred_pc1_gt
            cond = layer(cond)
        if one_name == 'pc1s':
            cond_pc1 = self.seperate_pc1(cond)
            cond_ori_pc1 = self.seperate_pc1(cond_ori)
        elif one_name == 'pc0s':
            cond_pc1 = self.seperate_pc0(cond)
            cond_ori_pc1 = self.seperate_pc0(cond_ori)


        for batch_idx in range(batch_sizes):#因为每个batch的噪声强度不同，所以分开处理
            mask = (cond_pc1.indices[:,0] == batch_idx)
            assert mask.shape[0] == cond_pc1.features.shape[0]
            mask_feature = cond_pc1.features[mask,:]
            mask_feature_ori = cond_ori_pc1.features[mask,:] #! add ori voxel feature
            
            cond_time.append(torch.cat([beta[batch_idx].unsqueeze(0).repeat(mask_feature.shape[0], 1), mask_feature, mask_feature_ori], dim=-1)) #每个batch添加对应的时间强度特征t
        # cond = cond.replace_feature(torch.cat(cond_time, dim=0))  # N,20 
        cond_time_cat = torch.cat(cond_time, dim=0)

        # cond_dimension = torch.full((cond_process.indices.shape[0], 1), 0, dtype=torch.int32, device='cuda')
        # cond_process_indice = torch.cat((cond_process.indices,cond_dimension), dim =1)
        if  isinstance(voxel_feature_pc1_noise, list):
            voxel_feature_pc1_noise = torch.cat(voxel_feature_pc1_noise, dim= 0)
        voxel_mix = torch.cat((cond_time_cat, voxel_feature_pc1_noise), dim = 1)
        # voxel_mix_coors = voxel_coor_pc1[:, :4]
        # voxel_mix_sparse4d = spconv.SparseConvTensor(voxel_mix.contiguous(), voxel_coor_pc1.contiguous(), spatial_shape + [2], batch_sizes)
        # out = voxel_mix_sparse4d
        voxel_mix_sparse3d = spconv.SparseConvTensor(voxel_mix.contiguous(), voxel_coor_pc1[:, :4].contiguous(), spatial_shape[:3], batch_sizes)
        out = self.pred_noise(voxel_mix_sparse3d)
        # for i, layer in enumerate(self.pred_noise): #pred_pc1_gt
        #     out = layer(out)
        # pred_pc1 = self.pred_pc1(out)
        # batch_size = coords.size(0)
        # beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        # cond = cond.view(batch_size, 1, -1)         # (B, 1, F)

        # time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        # cond_emb = torch.cat([time_emb, cond], dim=-1)    # (B, 1, F+3)
        
        # out = coords
        # for i, layer in enumerate(self.layers):
        #     out = layer(fea=out, cond=cond_emb)
        #     if i < len(self.layers) - 1:
        #         out = self.act(out)

        # if self.residual:
        #     return coords + out
        # else:
        return out



class DenoisingNet_point_noise(nn.Module):

    def __init__(self, point_dim, cond_dims, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        # self.layers = nn.ModuleList([
        #     PCNet(3, 128, cond_dims+3),
        #     PCNet(128, 256, cond_dims+3),
        #     PCNet(256, 512, cond_dims+3),
        #     PCNet(512, 256, cond_dims+3),
        #     PCNet(256, 128, cond_dims+3),
        #     PCNet(128, 3, cond_dims+3)
        # ])
        self.pred_noise = Spatio_Block(in_filters=16+16+4+16, mid_filters=32, out_filters=16, indice_key="cond_spatial")
        self.cond_spatial = nn.ModuleList([
                            SpatioTemporal_Decomposition_Block(in_filters=16, mid_filters=16, out_filters=16, indice_key="pn1"),
                            SpatioTemporal_Decomposition_Block(in_filters=16, mid_filters=16, out_filters=16, indice_key="pn2"),])
        self.seperate_pc1 = Seperate_to_pc1()
        self.seperate_pc0 = Seperate_to_3D(num_frames = 2)

    def forward(self, voxel_feature_pc1_noise, voxel_coor_pc1, beta, cond, batch_sizes, one_name):
        """
        Args:
            coords:   Noise point clouds at timestep t, (B, N, 3).
            beta:     Time. (B, time_dim=4).
            cond:     Condition. (N, C=16).  
        """
        spatial_shape = cond.spatial_shape
        cond_time = []
        cond_ori = cond
        # cond_process = self.cond_spatial(cond)
        for i, layer in enumerate(self.cond_spatial): #pred_pc1_gt
            cond = layer(cond)
        if one_name == 'pc1_all_noise':
            cond_pc1 = self.seperate_pc1(cond)
            cond_ori_pc1 = self.seperate_pc1(cond_ori)
        elif one_name == 'pc0_all_noise':
            cond_pc1 = self.seperate_pc0(cond)
            cond_ori_pc1 = self.seperate_pc0(cond_ori)


        for batch_idx in range(batch_sizes):#因为每个batch的噪声强度不同，所以分开处理
            mask = (cond_pc1.indices[:,0] == batch_idx)
            assert mask.shape[0] == cond_pc1.features.shape[0]
            mask_feature = cond_pc1.features[mask,:]
            mask_feature_ori = cond_ori_pc1.features[mask,:] #! add ori voxel feature
            
            cond_time.append(torch.cat([beta[batch_idx].unsqueeze(0).repeat(mask_feature.shape[0], 1), mask_feature, mask_feature_ori], dim=-1)) #每个batch添加对应的时间强度特征t
        # cond = cond.replace_feature(torch.cat(cond_time, dim=0))  # N,20 
        cond_time_cat = torch.cat(cond_time, dim=0)

        voxel_feature_pc1_noise = torch.cat(voxel_feature_pc1_noise, dim= 0)
        voxel_mix = torch.cat((cond_time_cat, voxel_feature_pc1_noise), dim = 1)
        # voxel_mix_coors = voxel_coor_pc1[:, :4]
        # voxel_mix_sparse4d = spconv.SparseConvTensor(voxel_mix.contiguous(), voxel_coor_pc1.contiguous(), spatial_shape + [2], batch_sizes)
        # out = voxel_mix_sparse4d
        voxel_mix_sparse3d = spconv.SparseConvTensor(voxel_mix.contiguous(), voxel_coor_pc1[:, :4].contiguous(), spatial_shape[:3], batch_sizes)
        out = self.pred_noise(voxel_mix_sparse3d)

        return out





# Conditional Point Diffusion Model
class CPDM(nn.Module):
    def __init__(self, pseudo_image_dims, sample_step):
        super().__init__()
        # self.config = config
        self.cond_dims = 16 #self.config.generator_config.cond_dims 
        # self.net = DenoisingNet(point_dim=3, cond_dims=self.cond_dims, residual=True)
        self.net = DenoisingNet(point_dim=3, cond_dims=self.cond_dims, residual=True)
        self.var_sched = VarianceSchedule()

        self.interval_nums = 4
        self.voxel_spatial_shape = pseudo_image_dims
        self.maxpooling = spconv.SparseMaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.sample_step = sample_step
        self.ddim_sampling_eta = 0.01
        print("self.sample_step", self.sample_step)
    
    def get_loss(self, voxel_feats_list_batch_dict, voxel_coors_list_batch_dict, pc0_voxel_feat, cond, batch_sizes, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            voxel_feats_list_all:    features N*16 ;  
            voxel_coors_list_all:    N*5 batch_idx, x,y,z, time_idx
            motion_feat:   features N*16; coors N*4 batch_idx, x,y,z

        """

        # batch_size, _, point_dim = coords.size()

        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_sizes, self.interval_nums) #self.interval_nums, batch_size

        total_loss = 0
        # motion_feat = motion_feat.replace_feature(torch.cat([pc0_voxel_feat, motion_feat.features], 1)) #! condition
        voxel_feature_pc1, voxel_coor_pc1 = voxel_feats_list_batch_dict['pc1s'], voxel_coors_list_batch_dict['pc1s']
        voxel_feature_pc1_gt, voxel_coor_pc1_gt = voxel_feats_list_batch_dict['pc1s_gt'], voxel_coors_list_batch_dict['pc1s_gt']
        device = pc0_voxel_feat.device
        voxel_feats_sp_pc1 = torch.cat(voxel_feature_pc1, dim=0)
        voxel_feats_sp_pc1_gt = torch.cat(voxel_feature_pc1_gt, dim=0)
        sparse_tensor_4d_pc1 = spconv.SparseConvTensor(voxel_feats_sp_pc1.contiguous(), voxel_coor_pc1[:,:-1].contiguous(), 
                                                   self.voxel_spatial_shape[:-1], batch_sizes)
        sparse_tensor_4d_pc1_gt = spconv.SparseConvTensor(voxel_feats_sp_pc1_gt.contiguous(), voxel_coor_pc1_gt[:,:-1].contiguous(), 
                                            self.voxel_spatial_shape[:-1], batch_sizes)
        diff_gt_dense = sparse_tensor_4d_pc1_gt.dense() - sparse_tensor_4d_pc1.dense() #只在pc1s上添加噪声 B,C,X,Y,Z
        diff_gt_pc1 = diff_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :] #N,16
        # mask_pc1s_voxel_coors = spconv.SparseConvTensor(torch.ones((voxel_coor_pc1.shape[0],1), device= device).contiguous(), 
        #                                                 voxel_coor_pc1[:,:-1].contiguous(), self.voxel_spatial_shape[:-1], batch_sizes)
        # dilated_mask_pc1s_voxel_coors = self.maxpooling(mask_pc1s_voxel_coors)
        for i in range(self.interval_nums):
            t = ts[i].to(device)
            time_code = self.var_sched.time_mlp(t)
            voxel_feature_pc1_noise = []
            for batch_idx in range(batch_sizes):
                batch_mask = (voxel_coor_pc1[:, 0] == batch_idx)
                noise =  torch.randn_like(diff_gt_pc1[batch_mask], device= device).float()
                pc1_noise = self.var_sched.q_sample(x_start=diff_gt_pc1[batch_mask], t=t[batch_idx], noise = noise)
                pc1_delta_noise = voxel_feature_pc1[batch_idx] + pc1_noise
                voxel_feature_pc1_noise.append(pc1_delta_noise)
            #! condition 是两帧特征的拼接的sparse tensor
            pred_pc1_gt = self.net(voxel_feature_pc1_noise, voxel_coor_pc1, beta=time_code, cond=cond, batch_sizes = batch_sizes) 
            #! pc1_gt_coor进行监督
            # pred_pc1_gt_dense = pred_pc1_gt.dense() # b,c,x,y,z torch.Size([4, 16, 512, 512, 32])
            # pred_pc1_cat = pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1_gt[:,0],voxel_coor_pc1_gt[:,1],voxel_coor_pc1_gt[:,2],voxel_coor_pc1_gt[:,3], :]
            #! pc1_coor进行监督
            restore_pc1_gt = sparse_tensor_4d_pc1_gt.dense().permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :]
            restore_pc1 = pred_pc1_gt.features
            # pred_pc1_list = []
            # for idx in range(batch_sizes):
            #     coor_mask_pc1_gt = (voxel_coor_pc1_gt[:,0] == idx)
            #     coor_pc1_gt_idx = voxel_coor_pc1_gt[coor_mask_pc1_gt,:][:,:4]
            #     pred_pc1_gt_idx = pred_pc1_gt_dense.permute(0,2,3,4,1)[coor_pc1_gt_idx[:,0],coor_pc1_gt_idx[:,1],coor_pc1_gt_idx[:,2],coor_pc1_gt_idx[:,3], :]
            #     pred_pc1_list.append(pred_pc1_gt_idx)
            # pred_pc1_cat = torch.cat(pred_pc1_list, dim=0)
            # pc1_gt_cat = torch.cat(voxel_feature_pc1_gt, dim=0) #!
            # alphas_cumprod = self.var_sched.alphas_cumprod[t]
            # beta = self.var_sched.betas[t]
            # sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       # (B, 1, 1)
            # sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)   # (B, 1, 1)
            
            # noise = torch.randn_like(coords)  # (B, N, d)
            # pred_noise = self.net(sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise, beta=beta, cond=cond)

            # loss = F.mse_loss(voxel_feats_sp_pc1_gt, pred_pc1_cat, reduction='mean')
            loss = F.mse_loss(restore_pc1_gt, restore_pc1, reduction='mean') #! change loss
            loss_pc1_feature1 = 1.0 - F.cosine_similarity(restore_pc1_gt, restore_pc1, dim=1).mean()
            total_loss += ((loss+loss_pc1_feature1) * (1.0 / self.interval_nums))


        return total_loss, restore_pc1
    
    def get_loss_multi(self, voxel_feats_list_batch_dict, voxel_coors_list_batch_dict, pc0_voxel_feat, cond, batch_sizes, voxel_feats_more, voxel_coors_more, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            voxel_feats_list_all:    features N*16 ;  
            voxel_coors_list_all:    N*5 batch_idx, x,y,z, time_idx
            motion_feat:   features N*16; coors N*4 batch_idx, x,y,z

        """

        # batch_size, _, point_dim = coords.size()
        total_loss = 0
        rand_num = random.randint(0, 3)
        for one, multi in (('pc0s','pc0_all'),('pc1s','pc1_all')):
    
            if ts == None: #!pc0和pc1加同样的噪声强度?
                ts = self.var_sched.recurrent_uniform_sampling(batch_sizes, self.interval_nums) #self.interval_nums, batch_size

            # motion_feat = motion_feat.replace_feature(torch.cat([pc0_voxel_feat, motion_feat.features], 1)) #! condition
            voxel_feature_pc1, voxel_coor_pc1 = voxel_feats_list_batch_dict[one], voxel_coors_list_batch_dict[one]
            voxel_feature_pc1_gt, voxel_coor_pc1_gt = voxel_feats_more[multi], voxel_coors_more[multi]
            device = pc0_voxel_feat.device
            voxel_feats_sp_pc1 = torch.cat(voxel_feature_pc1, dim=0)
            voxel_feats_sp_pc1_gt = torch.cat(voxel_feature_pc1_gt, dim=0)
            sparse_tensor_4d_pc1 = spconv.SparseConvTensor(voxel_feats_sp_pc1.contiguous(), voxel_coor_pc1[:,:-1].contiguous(), 
                                                    self.voxel_spatial_shape[:-1], batch_sizes)
            sparse_tensor_4d_pc1_gt = spconv.SparseConvTensor(voxel_feats_sp_pc1_gt.contiguous(), voxel_coor_pc1_gt[:,:-1].contiguous(), 
                                                self.voxel_spatial_shape[:-1], batch_sizes)
            diff_gt_dense = sparse_tensor_4d_pc1_gt.dense() - sparse_tensor_4d_pc1.dense() #只在pc1s上添加噪声 B,C,X,Y,Z
            diff_gt_pc1 = diff_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :] #N,16
            # mask_pc1s_voxel_coors = spconv.SparseConvTensor(torch.ones((voxel_coor_pc1.shape[0],1), device= device).contiguous(), 
            #                                                 voxel_coor_pc1[:,:-1].contiguous(), self.voxel_spatial_shape[:-1], batch_sizes)
            # dilated_mask_pc1s_voxel_coors = self.maxpooling(mask_pc1s_voxel_coors)
            restore_pc_list = []
            for i in range(self.interval_nums):
                t = ts[i].to(device)
                time_code = self.var_sched.time_mlp(t)
                voxel_feature_pc1_noise = []
                for batch_idx in range(batch_sizes):
                    batch_mask = (voxel_coor_pc1[:, 0] == batch_idx)
                    noise =  torch.randn_like(diff_gt_pc1[batch_mask], device= device).float()
                    pc1_noise = self.var_sched.q_sample(x_start=diff_gt_pc1[batch_mask], t=t[batch_idx], noise = noise)
                    # pc1_delta_noise = voxel_feature_pc1[batch_idx] + pc1_noise #! change to only noise
                    pc1_delta_noise = pc1_noise
                    voxel_feature_pc1_noise.append(pc1_delta_noise)
                #! condition 是两帧特征的拼接的sparse tensor cond添加原始稀疏帧的特征
                pred_pc1_gt = self.net(voxel_feature_pc1_noise, voxel_coor_pc1, beta=time_code, 
                                       cond=cond, batch_sizes = batch_sizes, one_name = one) 
                #! pc1_gt_coor进行监督
                # pred_pc1_gt_dense = pred_pc1_gt.dense() # b,c,x,y,z torch.Size([4, 16, 512, 512, 32])
                # pred_pc1_cat = pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1_gt[:,0],voxel_coor_pc1_gt[:,1],voxel_coor_pc1_gt[:,2],voxel_coor_pc1_gt[:,3], :]
                #! pc1_coor进行监督 
                # # TODO only diff 监督
                # restore_pc1_gt = sparse_tensor_4d_pc1_gt.dense().permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :]
                restore_pc1_delta = pred_pc1_gt.features
                restore_pc_list.append(restore_pc1_delta + torch.cat(voxel_feature_pc1, dim = 0))

                # pred_pc1_list = []
                # for idx in range(batch_sizes):
                #     coor_mask_pc1_gt = (voxel_coor_pc1_gt[:,0] == idx)
                #     coor_pc1_gt_idx = voxel_coor_pc1_gt[coor_mask_pc1_gt,:][:,:4]
                #     pred_pc1_gt_idx = pred_pc1_gt_dense.permute(0,2,3,4,1)[coor_pc1_gt_idx[:,0],coor_pc1_gt_idx[:,1],coor_pc1_gt_idx[:,2],coor_pc1_gt_idx[:,3], :]
                #     pred_pc1_list.append(pred_pc1_gt_idx)
                # pred_pc1_cat = torch.cat(pred_pc1_list, dim=0)
                # pc1_gt_cat = torch.cat(voxel_feature_pc1_gt, dim=0) #!
                # alphas_cumprod = self.var_sched.alphas_cumprod[t]
                # beta = self.var_sched.betas[t]
                # sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       # (B, 1, 1)
                # sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)   # (B, 1, 1)
                
                # noise = torch.randn_like(coords)  # (B, N, d)
                # pred_noise = self.net(sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise, beta=beta, cond=cond)

                # loss = F.mse_loss(voxel_feats_sp_pc1_gt, pred_pc1_cat, reduction='mean')
                loss = F.mse_loss(diff_gt_pc1, restore_pc1_delta, reduction='mean') #! change loss
                loss_pc1_feature1 = 1.0 - F.cosine_similarity(diff_gt_pc1, restore_pc1_delta, dim=1).mean()
                total_loss += ((loss+loss_pc1_feature1) * (1.0 / self.interval_nums))
            if one == 'pc0s':
                restore_pc_0 = restore_pc_list[rand_num]
            elif one == 'pc1s':
                restore_pc_1 = restore_pc_list[rand_num]
            


        return total_loss, restore_pc_0, restore_pc_1






    def val_restore_multi(self, voxel_feats_list_batch_dict, voxel_coors_list_batch_dict, pc0_voxel_feat, cond, batch_sizes, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            voxel_feats_list_all:    features N*16 ;  
            voxel_coors_list_all:    N*5 batch_idx, x,y,z, time_idx
            motion_feat:   features N*16; coors N*4 batch_idx, x,y,z

        """
        
        times = torch.linspace(-1, 999, steps=self.sample_step + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        device = pc0_voxel_feat.device
        # batch_size, _, point_dim = coords.size()
        for one in (('pc0s'),('pc1s')):
            voxel_feature_pc1, voxel_coor_pc1 = voxel_feats_list_batch_dict[one], voxel_coors_list_batch_dict[one] #原始稀疏特征
            voxel_feature_pc1_noise = []
            for batch_idx in range(batch_sizes):
                noise =  torch.randn_like(voxel_feature_pc1[batch_idx], device= device).float()
                pc1_delta_noise = noise
                voxel_feature_pc1_noise.append(pc1_delta_noise)
            

            for time, time_next in time_pairs:
            #self.interval_nums, batch_size
                ts = torch.full((batch_sizes,), time)
                # total_loss = 0
                # motion_feat = motion_feat.replace_feature(torch.cat([pc0_voxel_feat, motion_feat.features], 1)) #! condition
                
                t = ts.to(device)
                time_code = self.var_sched.time_mlp(t)
                #! condition 是两帧特征的拼接的sparse tensor
                pred_pc1_gt = self.net(voxel_feature_pc1_noise, voxel_coor_pc1, beta=time_code, cond=cond, batch_sizes = batch_sizes, one_name = one) 
                # pred_pc1_gt_dense = pred_pc1_gt.dense() # b,c,x,y,z torch.Size([4, 16, 512, 512, 32])

                # pred_pc1_cat = pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1_gt[:,0],voxel_coor_pc1_gt[:,1],voxel_coor_pc1_gt[:,2],voxel_coor_pc1_gt[:,3], :]
                restore_pc1 = pred_pc1_gt.features   #pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :]
                if time_next < 0:
                    continue
                pred_noise = self.var_sched.predict_noise_from_start(torch.cat(voxel_feature_pc1, dim = 0), t, restore_pc1)
                voxel_feature_pc1_noise = self.var_sched.reconstruction(time, time_next, restore_pc1, pred_noise)
            if one == 'pc0s':
                voxel_feature_pc_0 = torch.cat(voxel_feature_pc1, dim = 0)
                restore_pc_0 = restore_pc1 + voxel_feature_pc_0
            elif one == 'pc1s':
                voxel_feature_pc_1 = torch.cat(voxel_feature_pc1, dim = 0)
                restore_pc_1 = restore_pc1 + voxel_feature_pc_1
        return restore_pc_0, restore_pc_1




    def val_restore(self, voxel_feats_list_batch_dict, voxel_coors_list_batch_dict, pc0_voxel_feat, cond, batch_sizes, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            voxel_feats_list_all:    features N*16 ;  
            voxel_coors_list_all:    N*5 batch_idx, x,y,z, time_idx
            motion_feat:   features N*16; coors N*4 batch_idx, x,y,z

        """

        # batch_size, _, point_dim = coords.size()

 #self.interval_nums, batch_size
        ts = torch.full((batch_sizes,), 1000.0)
        # total_loss = 0
        # motion_feat = motion_feat.replace_feature(torch.cat([pc0_voxel_feat, motion_feat.features], 1)) #! condition
        voxel_feature_pc1, voxel_coor_pc1 = voxel_feats_list_batch_dict['pc1s'], voxel_coors_list_batch_dict['pc1s']
        device = pc0_voxel_feat.device


        t = ts.to(device)
        time_code = self.var_sched.time_mlp(t)
        voxel_feature_pc1_noise = []
        for batch_idx in range(batch_sizes):
            # batch_mask = (voxel_coor_pc1[:, 0] == batch_idx)
            noise =  torch.randn_like(voxel_feature_pc1[batch_idx], device= device).float()
            # pc1_noise = self.var_sched.q_sample(x_start=diff_gt_pc1[batch_mask], t=t[batch_idx], noise = noise)
            pc1_delta_noise = voxel_feature_pc1[batch_idx] + noise
            voxel_feature_pc1_noise.append(pc1_delta_noise)
        #! condition 是两帧特征的拼接的sparse tensor
        pred_pc1_gt = self.net(voxel_feature_pc1_noise, voxel_coor_pc1, beta=time_code, cond=cond, batch_sizes = batch_sizes) 
        pred_pc1_gt_dense = pred_pc1_gt.dense() # b,c,x,y,z torch.Size([4, 16, 512, 512, 32])

        # pred_pc1_cat = pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1_gt[:,0],voxel_coor_pc1_gt[:,1],voxel_coor_pc1_gt[:,2],voxel_coor_pc1_gt[:,3], :]
        restore_pc1 = pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :]



        return restore_pc1
    


    def get_loss_point_noise(self, voxel_feats_list_batch_dict, voxel_coors_list_batch_dict, pc0_voxel_feat, cond, batch_sizes, add_noise_dict):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            voxel_feats_list_all:    features N*16 ;  
            voxel_coors_list_all:    N*5 batch_idx, x,y,z, time_idx
            motion_feat:   features N*16; coors N*4 batch_idx, x,y,z

        """

        # batch_size, _, point_dim = coords.size()
        total_loss = 0

        for one, multi in (('pc0_all_noise','pc0_all'),('pc1_all_noise','pc1_all')):
    
            # motion_feat = motion_feat.replace_feature(torch.cat([pc0_voxel_feat, motion_feat.features], 1)) #! condition
            voxel_feature_pc1, voxel_coor_pc1 = voxel_feats_list_batch_dict[one], voxel_coors_list_batch_dict[one]
            voxel_feature_pc1_gt, voxel_coor_pc1_gt = voxel_feats_list_batch_dict[multi], voxel_coors_list_batch_dict[multi]
            # device = pc0_voxel_feat.device
            # voxel_feats_sp_pc1 = torch.cat(voxel_feature_pc1, dim=0)
            voxel_feats_sp_pc1_gt = torch.cat(voxel_feature_pc1_gt, dim=0)
            # sparse_tensor_4d_pc1 = spconv.SparseConvTensor(voxel_feats_sp_pc1.contiguous(), voxel_coor_pc1[:,:-1].contiguous(), 
            #                                         self.voxel_spatial_shape[:-1], batch_sizes)
            sparse_tensor_4d_pc1_gt = spconv.SparseConvTensor(voxel_feats_sp_pc1_gt.contiguous(), voxel_coor_pc1_gt[:,:-1].contiguous(), 
                                                self.voxel_spatial_shape[:-1], batch_sizes)
            # diff_gt_dense = sparse_tensor_4d_pc1_gt.dense() - sparse_tensor_4d_pc1.dense() #只在pc1s上添加噪声 B,C,X,Y,Z
            # diff_gt_pc1 = diff_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :] #N,16
            # mask_pc1s_voxel_coors = spconv.SparseConvTensor(torch.ones((voxel_coor_pc1.shape[0],1), device= device).contiguous(), 
            #                                                 voxel_coor_pc1[:,:-1].contiguous(), self.voxel_spatial_shape[:-1], batch_sizes)
            # dilated_mask_pc1s_voxel_coors = self.maxpooling(mask_pc1s_voxel_coors)


            #! condition 是两帧特征的拼接的sparse tensor cond添加原始稀疏帧的特征
            pred_pc1_gt = self.net(voxel_feature_pc1, voxel_coor_pc1, beta=time_code, 
                                    cond=cond, batch_sizes = batch_sizes, one_name = one) 
            #! pc1_gt_coor进行监督
            # pred_pc1_gt_dense = pred_pc1_gt.dense() # b,c,x,y,z torch.Size([4, 16, 512, 512, 32])
            # pred_pc1_cat = pred_pc1_gt_dense.permute(0,2,3,4,1)[voxel_coor_pc1_gt[:,0],voxel_coor_pc1_gt[:,1],voxel_coor_pc1_gt[:,2],voxel_coor_pc1_gt[:,3], :]
            #! pc1_coor进行监督
            restore_pc1_gt = sparse_tensor_4d_pc1_gt.dense().permute(0,2,3,4,1)[voxel_coor_pc1[:,0],voxel_coor_pc1[:,1],voxel_coor_pc1[:,2],voxel_coor_pc1[:,3], :]
            restore_pc1 = pred_pc1_gt.features


            loss = F.mse_loss(restore_pc1_gt, restore_pc1, reduction='mean') #! change loss
            loss_pc1_feature1 = 1.0 - F.cosine_similarity(restore_pc1_gt, restore_pc1, dim=1).mean()

            total_loss += ((loss+loss_pc1_feature1) * (1.0 / self.interval_nums))
            if one == 'pc0_all_noise':
                restore_pc_0 = restore_pc1
            elif one == 'pc1_all_noise':
                restore_pc_1 = restore_pc1
            


        return total_loss, restore_pc_0, restore_pc_1
    

