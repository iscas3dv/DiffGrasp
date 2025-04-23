# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import mano
from data.process_Arctic_data_v4 import tensor_destandardization, tensor_standardization
from tools.utils import makepath, to_cpu, to_np, to_tensor, create_video

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from bps_torch.bps import bps_torch
import chamfer_distance as chd
from psbody.mesh import Mesh
from manopth.manolayer import ManoLayer
from tools.vis_tools import sp_animation
class DiffOptim(nn.Module):
    
    def __init__(self,
                 bs,
                 fs,
                 obj_sdfs,
                 ds_info,
                 cfg=None):
        super(DiffOptim, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg

        self.bs = bs
        self.fs = fs
        self.obj_sdfs = obj_sdfs
        
        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')
        self.lr = 1e-4
        mano_root = '/home/ljh/disk_4T/zyh/GOAL/manopth/mano/models'
        self.rh_mano = ManoLayer(mano_root=mano_root,
                                use_pca=False, 
                                ncomps=45,
                                joint_rot_mode='rotmat',
                                side='right').to(self.device)
        
        self.lh_mano = ManoLayer(mano_root=mano_root, 
                                use_pca=False, 
                                ncomps=45,
                                joint_rot_mode='rotmat',
                                side='left').to(self.device)
        self.ds_info = ds_info
        
    def prepare_recon(self, batch, sampled_results, is_world=True):
        
        recon_output = {}
        
        bs = self.bs
        fs = batch['lh_transl'].shape[1]   
        
        lh_betas = batch['lh_betas']
        rh_betas = batch['rh_betas']
        
        # lh_norm_pose = self.ds_info['lh']
        # rh_norm_pose = self.ds_info['rh']

        # sampled_results['rh_transl'] = tensor_destandardization(sampled_results['rh_transl'], 
        #                                               rh_norm_pose['transl_mean'].to(self.device), rh_norm_pose['transl_std'].to(self.device))
        # sampled_results['rh_pose_rotmat'] = tensor_destandardization(sampled_results['rh_pose_rotmat'], 
        #                                                    rh_norm_pose['pose_mean'][:,:2,:].to(self.device).reshape(-1), rh_norm_pose['pose_std'][:,:2,:].to(self.device).reshape(-1))
        # sampled_results['lh_transl'] = tensor_destandardization(sampled_results['lh_transl'], 
        #                                               lh_norm_pose['transl_mean'].to(self.device), lh_norm_pose['transl_std'].to(self.device))
        # sampled_results['lh_pose_rotmat'] = tensor_destandardization(sampled_results['lh_pose_rotmat'], 
        #                                                    lh_norm_pose['pose_mean'][:,:2,:].to(self.device).reshape(-1), lh_norm_pose['pose_std'][:,:2,:].to(self.device).reshape(-1))

        lh_pred_vert, lh_pred_joints = self.manoLayer(poses=sampled_results['lh_pose_rotmat'][:bs,...],
                                 trans=sampled_results['lh_transl'][:bs,...], betas= lh_betas[:bs,...], is_right=False, is_6D=True)
        rh_pred_vert, rh_pred_joints = self.manoLayer(poses=sampled_results['rh_pose_rotmat'][:bs,...],
                                 trans=sampled_results['rh_transl'][:bs,...], betas= rh_betas[:bs,...], is_right=True, is_6D=True)

        ### hands verts&joints from obj to world coords
        if is_world:
            obj2world_rot = batch['obj_orient_rotmat'][:bs,...].reshape(bs*fs, 3, 3).to(self.device)
            obj2world_transl = batch['obj_transl'][:bs,...].reshape(bs*fs, 3).to(self.device)
            lh_pred_vert = torch.matmul(lh_pred_vert.unsqueeze(2), obj2world_rot.permute(0,2,1).unsqueeze(1)).squeeze() + obj2world_transl.unsqueeze(1)
            rh_pred_vert = torch.matmul(rh_pred_vert.unsqueeze(2), obj2world_rot.permute(0,2,1).unsqueeze(1)).squeeze() + obj2world_transl.unsqueeze(1)
            lh_pred_joints = torch.matmul(lh_pred_joints.unsqueeze(2), obj2world_rot.permute(0,2,1).unsqueeze(1)).squeeze() + obj2world_transl.unsqueeze(1)
            rh_pred_joints = torch.matmul(rh_pred_joints.unsqueeze(2), obj2world_rot.permute(0,2,1).unsqueeze(1)).squeeze() + obj2world_transl.unsqueeze(1)       

        
        recon_output['lh_pred_vert'] = lh_pred_vert.reshape(bs,fs,-1,3)
        recon_output['lh_pred_joints'] = lh_pred_joints.reshape(bs,fs,-1,3)
        recon_output['rh_pred_vert'] = rh_pred_vert.reshape(bs,fs,-1,3)
        recon_output['rh_pred_joints'] = rh_pred_joints.reshape(bs,fs,-1,3)
        
        return recon_output
    
    def manoLayer(self, poses, trans, betas, is_right, is_6D=True):
        
        bs = poses.shape[0]
        fs = poses.shape[1]
        params = {}
        if is_6D:            
            poses = d62rotmat(poses).reshape([bs*fs, -1, 3, 3])
            
        poses = poses.reshape([bs*fs, -1, 3, 3]).to(self.device)
        poses = rotmat2aa(poses).reshape([bs*fs,48])
        
        params['hand_pose'] = poses[:,3:]
        params['betas'] = betas.reshape(bs*fs, 10).to(self.device)
        params['transl'] = trans.reshape(bs*fs, 3).to(self.device)
        params['global_orient'] = poses[:,0:3]
        
        if is_right:
            mano_layer = mano.load(model_path='/home/ljh/disk_4T/zyh/GOAL/MANO/models/mano/MANO_RIGHT.pkl',
                                model_type='mano',
                                use_pca=False,
                                num_pca_comps=45,
                                batch_size=bs*fs,
                                flat_hand_mean=False).to(self.device)
        else:
            mano_layer = mano.load(model_path='/home/ljh/disk_4T/zyh/GOAL/MANO/models/mano/MANO_LEFT.pkl',
                                model_type='mano',
                                use_pca=False,
                                is_rhand=False,
                                num_pca_comps=45,
                                batch_size=bs*fs,
                                flat_hand_mean=False).to(self.device)

        ### keys: th_pose_coeffs, th_betas, th_trans
        mano_out = mano_layer(**params)
        vertices = mano_out.vertices
        joints = mano_out.joints
        # torch.Size([30, 778, 3]) torch.Size([30, 16, 3])
        return vertices, joints
    
    def fitting(self, batch, net_output):
        with torch.enable_grad():
            net_output = net_output.squeeze(1).permute(0,2,1)
            # torch.Size([bs, 30, 198])
            net_outdict = {'rh_transl': net_output[..., :3], 
                    'rh_pose_rotmat': net_output[..., 3:99],
                    'lh_transl': net_output[..., 99:102],
                    'lh_pose_rotmat': net_output[..., 102:198]}
            
            # 先反归一化出来
            lh_norm_pose = self.ds_info['lh']
            rh_norm_pose = self.ds_info['rh']
            net_outdict['rh_transl'] = tensor_destandardization(net_outdict['rh_transl'], 
                                                        rh_norm_pose['transl_mean'].to(self.device), rh_norm_pose['transl_std'].to(self.device))
            net_outdict['rh_pose_rotmat'] = tensor_destandardization(net_outdict['rh_pose_rotmat'], 
                                                            rh_norm_pose['pose_mean'][:,:2,:].to(self.device).reshape(-1), rh_norm_pose['pose_std'][:,:2,:].to(self.device).reshape(-1))
            net_outdict['lh_transl'] = tensor_destandardization(net_outdict['lh_transl'], 
                                                        lh_norm_pose['transl_mean'].to(self.device), lh_norm_pose['transl_std'].to(self.device))
            net_outdict['lh_pose_rotmat'] = tensor_destandardization(net_outdict['lh_pose_rotmat'], 
                                                           lh_norm_pose['pose_mean'][:,:2,:].to(self.device).reshape(-1), lh_norm_pose['pose_std'][:,:2,:].to(self.device).reshape(-1))
            
            recon_results_before = self.prepare_recon(batch, net_outdict, is_world=True)
            
            self.opt_params = {'rh_transl': nn.Parameter(net_outdict['rh_transl']), 
                    'rh_pose_rotmat': nn.Parameter(net_outdict['rh_pose_rotmat']),
                    'lh_transl': nn.Parameter(net_outdict['lh_transl']),
                    'lh_pose_rotmat': nn.Parameter(net_outdict['lh_pose_rotmat'])}
            
            # self.opt_params = {k:net_outdict[k][:self.bs,...].clone().requires_grad_(True) for k in net_outdict.keys()}
            
            # TODO: 修改可优化参数/调整参数的weight_decay
            # self.opt_s3 = optim.Adam([self.opt_params[k] for k in self.opt_params.keys()], lr=self.lr)
            self.opt_s3 = optim.Adam([self.opt_params[k] for k in ['rh_pose_rotmat', 'lh_pose_rotmat']], lr=self.lr)
            # self.opt_s3 = optim.Adam([self.opt_params[k] for k in ['lh_transl', 'rh_transl']], lr=self.lr)

            
            # recon_return = self.prepare_recon(batch, net_outdict, is_world=True)

            self.optimizers = [self.opt_s3]
            self.num_iters = [200]

            for stg, optimizer in enumerate(self.optimizers):
                for itr in range(self.num_iters[stg]):
                    optimizer.zero_grad()
                    losses = self.calc_loss(batch)
                    losses['loss_total'].backward()
                    optimizer.step()
                    # print(self.opt_params)
                    if itr % 10 == 0:
                        print(self.create_loss_message(losses, stg, itr))
            
            ### for vis only
            opt_vis = {k:v.detach().clone() for k,v in self.opt_params.items()}
            # recon_results = self.prepare_recon(batch, opt_vis, is_world=True)
            
            # fs = batch['obj_sdf_idx'].shape[1]
            
            # mesh_path = '/home/ljh/disk_4T/dataset/arctic_split_m1_max/val/obj_mesh'
            # obj_sdf_idx = to_cpu(batch['obj_sdf_idx'][0,0])
            # mesh_path = os.path.join(mesh_path, 'fixed_obj_%s.obj'%str(obj_sdf_idx))
            # from psbody.mesh.colors import name_to_rgb
            # obj_fixed_mesh = Mesh(filename=mesh_path)
            # obj_fixed_mesh.vc = name_to_rgb['yellow']
            # # obj_fixed_mesh.v = to_cpu(torch.matmul(torch.tensor(obj_fixed_mesh.v), batch['obj_orient_rotmat'][0].permute(0,2,1).to(self.device)) + batch['obj_transl'][0].unsqueeze(1).to(self.device))
            
            # # obj_verts
            # # print(obj_fixed_mesh.v)
            # obj_verts_for_vis = torch.tensor(obj_fixed_mesh.v.reshape(1,-1,3).repeat(fs, axis=0)).to(self.device).to(torch.float32)
            # obj_verts_for_vis = torch.matmul(obj_verts_for_vis, batch['obj_orient_rotmat'][0].permute(0,2,1).to(self.device)) + batch['obj_transl'][0].unsqueeze(1).to(self.device)
            
            # before_dict = torch.cat([v.reshape(self.bs, self.fs,-1).detach().clone() for v in net_outdict.values()], dim=2)
            # opt_dict = torch.cat([v.reshape(self.bs, self.fs,-1).detach().clone() for v in opt_vis.values()], dim=2)
            
            # sumup_results = (before_dict + opt_dict) / 2
            # # torch.Size([bs, 30, 198])
            # sumup_outdict = {'rh_transl': sumup_results[..., :3], 
            #         'rh_pose_rotmat': sumup_results[..., 3:99],
            #         'lh_transl': sumup_results[..., 99:102],
            #         'lh_pose_rotmat': sumup_results[..., 102:198]}
            # sumup_results = self.prepare_recon(batch, sumup_outdict, is_world=True)
            
            # sp_anim = sp_animation()
            # for i in range(fs):
            #     obj_mesh = Mesh(v=to_cpu(obj_verts_for_vis)[i], f=obj_fixed_mesh.f,vc=name_to_rgb['yellow'])
            #     lh_mesh = Mesh(v=to_cpu(recon_results['lh_pred_vert'][0][i]), f=to_cpu(self.lh_mano.th_faces), vc=name_to_rgb['green'])
            #     rh_mesh = Mesh(v=to_cpu(recon_results['rh_pred_vert'][0][i]), f=to_cpu(self.rh_mano.th_faces), vc=name_to_rgb['green'])
            #     lh_mesh_b = Mesh(v=to_cpu(recon_results_before['lh_pred_vert'][0][i]), f=to_cpu(self.lh_mano.th_faces), vc=name_to_rgb['pink'])
            #     rh_mesh_b = Mesh(v=to_cpu(recon_results_before['rh_pred_vert'][0][i]), f=to_cpu(self.rh_mano.th_faces), vc=name_to_rgb['pink'])
            #     lh_mesh_s = Mesh(v=to_cpu(sumup_results['lh_pred_vert'][0][i]), f=to_cpu(self.lh_mano.th_faces), vc=name_to_rgb['blue'])
            #     rh_mesh_s = Mesh(v=to_cpu(sumup_results['rh_pred_vert'][0][i]), f=to_cpu(self.rh_mano.th_faces), vc=name_to_rgb['blue'])
            #     sp_anim.add_frame([obj_mesh, lh_mesh, rh_mesh, lh_mesh_b, rh_mesh_b, lh_mesh_s, rh_mesh_s], 
            #                       ['obj_mesh', 'lh_mesh', 'rh_mesh', 'lh_mesh_b', 'rh_mesh_b', 'lh_mesh_s', 'rh_mesh_s'])               
            # sp_anim.save_animation('./vis.html')
            # exit()
            # return之前记得要standerization回去
            opt_vis['rh_transl'] = tensor_standardization(opt_vis['rh_transl'],
                                                        rh_norm_pose['transl_mean'].to(self.device), rh_norm_pose['transl_std'].to(self.device))
            
            opt_vis['rh_pose_rotmat'] = tensor_standardization(opt_vis['rh_pose_rotmat'],
                                                    rh_norm_pose['pose_mean'][...,:2,:].reshape(1,1,-1).to(self.device), rh_norm_pose['pose_std'][...,:2,:].reshape(1,1,-1).to(self.device))
            
            opt_vis['lh_transl'] = tensor_standardization(opt_vis['lh_transl'],
                                                        lh_norm_pose['transl_mean'].to(self.device), lh_norm_pose['transl_std'].to(self.device))
            
            opt_vis['lh_pose_rotmat'] = tensor_standardization(opt_vis['lh_pose_rotmat'],
                                                    lh_norm_pose['pose_mean'][...,:2,:].reshape(1,1,-1).to(self.device), lh_norm_pose['pose_std'][...,:2,:].reshape(1,1,-1).to(self.device))
                
            opt_results = {}
            opt_results['img'] = torch.cat([v.reshape(self.bs, self.fs,-1).detach().clone() for v in opt_vis.values()], dim=2).permute(0,2,1).unsqueeze(1)

            return opt_results

    def calc_loss(self, batch):
        # ###### 注意！！目前bs只能为1
        # obj_sdf_idx = batch['obj_sdf_idx'][:self.bs,0]
        # obj_sdf_idx = to_cpu(obj_sdf_idx).tolist()
        # sdfs = torch.tensor(self.obj_sdfs[obj_sdf_idx,...]).to(self.device) #torch.Size([bs, 128, 128, 128, 4])
        # sdfs_value = sdfs[...,0]
        # sdfs_grids = sdfs[...,1:4]
        # ### 手先变换到物体坐标系下
        # recon_out_objcoords = self.prepare_recon(batch, self.opt_params, is_world=False)     
        # ### sdf的grid坐标是在物体坐标系下的值, 将grid和手点云变换到以0为起点
        # bbmin = torch.min(sdfs_grids.reshape(self.bs,-1,3),dim=1).values
        # bbmax = torch.max(sdfs_grids.reshape(self.bs,-1,3),dim=1).values
        # sdfs_zerone = sdfs_grids-bbmin
        # lr_verts_zerone = torch.concatenate((recon_out_objcoords['lh_pred_vert'],recon_out_objcoords['rh_pred_vert']),dim=-2)-bbmin
        # ### 看看手上的点距离点云有多少个就知道是哪个子网格
        # sdfs_strides = torch.mean((bbmax-bbmin)/128,dim=1)
        # lr_grid_idx = lr_verts_zerone//sdfs_strides
        
        # mask = torch.sum(torch.logical_and(lr_grid_idx>=0,lr_grid_idx<=126),dim=-1) == 3
        # ### 得到点
        # valid_lr_verts = lr_verts_zerone[mask]
        # ### 得到点所在的网格八顶点
        # lr_grid_idx[mask]
        # ### 知道正负号
        fs = batch['obj_sdf_idx'].shape[1]
        ### 获得物体的mesh(目前所有操作都是在物体坐标系下)
        mesh_path = '/home/ljh/disk_4T/dataset/arctic_split_m1_max/val/obj_mesh'
        obj_sdf_idx = to_cpu(batch['obj_sdf_idx'][0,0])
        mesh_path = os.path.join(mesh_path, 'fixed_obj_%s.obj'%str(obj_sdf_idx))
        from psbody.mesh.colors import name_to_rgb
        obj_fixed_mesh = Mesh(filename=mesh_path)
        obj_fixed_mesh.vc = name_to_rgb['yellow']
        # obj_fixed_mesh.v = to_cpu(torch.matmul(torch.tensor(obj_fixed_mesh.v), batch['obj_orient_rotmat'][0].permute(0,2,1).to(self.device)) + batch['obj_transl'][0].unsqueeze(1).to(self.device))
        
        # obj_verts
        # print(obj_fixed_mesh.v)
        obj_verts_for_vis = torch.tensor(obj_fixed_mesh.v.reshape(1,-1,3).repeat(fs, axis=0)).to(self.device).to(torch.float32)
        obj_verts_for_vis = torch.matmul(obj_verts_for_vis, batch['obj_orient_rotmat'][0].permute(0,2,1).to(self.device)) + batch['obj_transl'][0].unsqueeze(1).to(self.device)
        
        
        obj_norms_batch = torch.tensor(obj_fixed_mesh.estimate_vertex_normals().reshape(1,-1,3).repeat(fs*2, axis=0)).to(self.device).to(torch.float32)
        # obj_verts_batch = torch.tensor(obj_fixed_mesh.v.reshape(1,-1,3).repeat(fs*2, axis=0)).to(self.device).to(torch.float32)
        obj_verts_batch = obj_verts_for_vis.repeat(2,1,1)
        
        lh_norms = []
        rh_norms = []
        obj_norms = []

        recon_out_objcoords = self.prepare_recon(batch, self.opt_params, is_world=True)     
        
        sp_anim = sp_animation()
        for i in range(fs):
            obj_mesh = Mesh(v=to_cpu(obj_verts_for_vis)[i], f=obj_fixed_mesh.f,vc=name_to_rgb['yellow'])
            lh_mesh = Mesh(v=to_cpu(recon_out_objcoords['lh_pred_vert'][0][i]), f=to_cpu(self.lh_mano.th_faces), vc=name_to_rgb['green'])
            rh_mesh = Mesh(v=to_cpu(recon_out_objcoords['rh_pred_vert'][0][i]), f=to_cpu(self.rh_mano.th_faces), vc=name_to_rgb['green'])
            sp_anim.add_frame([obj_mesh, lh_mesh, rh_mesh], ['obj_mesh', 'lh_mesh', 'rh_mesh'])
            obj_norms.append(obj_mesh.estimate_vertex_normals())
            lh_norms.append(lh_mesh.estimate_vertex_normals())
            rh_norms.append(rh_mesh.estimate_vertex_normals())
            
        # sp_anim.save_animation('./vis.html')
        obj_norms = torch.from_numpy(np.array(obj_norms))
        obj_norms = obj_norms.repeat(2,1,1).to(self.device).to(torch.float32)
    
        lh_norms = torch.from_numpy(np.array(lh_norms))
        rh_norms = torch.from_numpy(np.array(rh_norms))
        
        lr_norms = torch.concatenate((lh_norms,rh_norms),dim=0).to(self.device).to(torch.float32)
        lr_verts = torch.concatenate((recon_out_objcoords['lh_pred_vert'][0], recon_out_objcoords['rh_pred_vert'][0])).to(torch.float32)
        
        y2x_signed, x2y_signed, yidx_near, xidx_near = self.point2point_signed(obj_verts_batch, lr_verts, obj_norms, lr_norms)

        # print(y2x_signed.shape)
        # print(torch.sum(y2x_signed<0,dim=1))
        # print(torch.sum(y2x_signed<0,dim=1))
        # exit()
        losses = {}
        ### 找到穿透的点并约束为0
        loss_weight_pene = 1.
        losses['peneration_loss'] = torch.abs(torch.sum(y2x_signed[y2x_signed < 0])) * loss_weight_pene
        
        ### 找到阈值小于某个值的点约束为0
        loss_weight_contact = 0.1
        threshold = 0.01
        losses['contact_loss'] = torch.abs(torch.sum(y2x_signed[torch.logical_and(y2x_signed>0, y2x_signed<threshold)])) * loss_weight_contact
        
        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total
        
        return losses 

        
    def point2point_signed(self, 
            x,
            y,
            x_normals=None,
            y_normals=None,
            return_vector=False,
    ):
        """
        signed distance between two pointclouds
        Args:
            x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
                with P1 points in each batch element, batch size N and feature
                dimension D.
            y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
                with P2 points in each batch element, batch size N and feature
                dimension D.
            x_normals: Optional FloatTensor of shape (N, P1, D).
            y_normals: Optional FloatTensor of shape (N, P2, D).
        Returns:
            - y2x_signed: Torch.Tensor
                the sign distance from y to x
            - y2x_signed: Torch.Tensor
                the sign distance from y to x
            - yidx_near: Torch.tensor
                the indices of x vertices closest to y
        """


        N, P1, D = x.shape
        P2 = y.shape[1]

        if y.shape[0] != N or y.shape[2] != D:
            raise ValueError("y does not have the correct shape.")

        ch_dist = chd.ChamferDistance()

        x_near, y_near, xidx_near, yidx_near = ch_dist(x,y,x_normals=x_normals,y_normals=y_normals)

        xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
        x_near = y.gather(1, xidx_near_expanded)

        yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
        y_near = x.gather(1, yidx_near_expanded)

        x2y = x - x_near  # y point to x
        y2x = y - y_near  # x point to y

        if x_normals is not None:
            y_nn = x_normals.gather(1, yidx_near_expanded)
            in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
            y2x_signed = y2x.norm(dim=2) * in_out

        else:
            y2x_signed = y2x.norm(dim=2)

        if y_normals is not None:
            x_nn = y_normals.gather(1, xidx_near_expanded)
            in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
            x2y_signed = x2y.norm(dim=2) * in_out_x
        else:
            x2y_signed = x2y.norm(dim=2)

        if not return_vector:
            return y2x_signed, x2y_signed, yidx_near, xidx_near
        else:
            return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y

    
    
    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'



class GNetOptim(nn.Module):

    def __init__(self,
                 sbj_model,
                 obj_model,
                 cfg,
                 verbose = False
                 ):
        super(GNetOptim, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg
        self.body_model_cfg = cfg.body_model

        self.sbj_m = sbj_model
        self.obj_m = obj_model

        self.config_optimizers()

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.rhand_idx = to_tensor(np.load(self.cfg.losses.rh2smplx_idx), dtype=torch.long)
        self.rhand_tri = to_tensor(np.load(self.cfg.losses.rh_faces).astype(np.int32))
        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)
        self.verbose = verbose

        self.bps_torch = bps_torch()
        self.ch_dist = chd.ChamferDistance()

    def config_optimizers(self):
        bs = 1
        self.bs = bs
        device = self.device
        dtype = self.dtype

        self.opt_params = {
            'global_orient'     : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'body_pose'         : torch.randn(bs, 21*3, device=device, dtype=dtype, requires_grad=True),
            'left_hand_pose'    : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=True),
            'right_hand_pose'   : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=True),
            'jaw_pose'          : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'leye_pose'         : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'reye_pose'         : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'transl'            : torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=True),
        }

        lr = self.cfg.get('smplx_opt_lr', 5e-3)
        # self.opt_s1 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl']], lr=lr)
        # self.opt_s2 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl', 'body_pose']], lr=lr)
        self.opt_s3 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl', 'body_pose', 'right_hand_pose']], lr=lr)
        
        self.optimizers = [self.opt_s3]

        self.num_iters = [200]

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')


    def init_params(self, start_params):

        fullpose_aa = rotmat2aa(start_params['fullpose_rotmat']).reshape(1, -1)

        start_params_aa = full2bone_aa(fullpose_aa, start_params['transl'])

        for k in self.opt_params.keys():
            self.opt_params[k].data = torch.repeat_interleave(start_params_aa[k], self.bs, dim=0)

    def get_smplx_verts(self, batch, output):

        B = batch['transl_obj'].shape[0]

        if batch['gender']==1:
            net_params = output['cnet']['m_params']
        else:
            net_params = output['cnet']['f_params']

        obj_params_gt = {'transl': batch['transl_obj'],
                         'global_orient': batch['global_orient_obj']}

        obj_output = self.obj_m(**obj_params_gt)

        self.obj_verts = obj_output.vertices
        self.sbj_params = net_params

        self.init_params(net_params)

        with torch.no_grad():
            sbj_output = self.sbj_m(**net_params)
            v = sbj_output.vertices.reshape(-1, 10475, 3)
            verts_sampled = v[:, self.verts_ids]

        return v, verts_sampled



    def calc_loss(self, batch, net_output, stage):


        opt_params = {k:aa2rotmat(v) for k,v in self.opt_params.items() if k!='transl'}
        opt_params['transl'] = self.opt_params['transl']

        output = self.sbj_m(**opt_params, return_full_pose = True)
        verts = output.vertices
        verts_sampled = verts[:,self.verts_ids]

        rh2obj = self.bps_torch.encode(x=self.obj_verts,
                                       feature_type=['deltas'],
                                       custom_basis=verts[:,self.verts_ids[self.rh_ids_sampled]])['deltas']

        rh2obj_net = net_output['cnet']['dist'].reshape(rh2obj.shape).detach()
        rh2obj_w = torch.exp(-5 * rh2obj_net.norm(dim=-1, keepdim=True))

        gaze_net = net_output['cnet']['gaze']
        gaze_opt_vec = verts_sampled[:, 386] - verts_sampled[:, 387]
        gaze_opt = gaze_opt_vec / gaze_opt_vec.norm(dim=-1, keepdim=True)

        losses = {
            "dist_rh2obj": 2*self.LossL1(rh2obj_w*rh2obj,rh2obj_w*rh2obj_net),
            "grnd_contact": (verts[:,:,1].min() < -.02)*torch.pow(verts[:,:,1].min()+.01, 2),
            "gaze": 1 * self.LossL1(gaze_net.detach(), gaze_opt),
            # 'penet': 1  *torch.pow(rh2obj_penet[is_penet], 2).mean()
        }



        body_loss = {k: self.LossL2(rotmat2aa(self.sbj_params[k]).detach().reshape(-1), self.opt_params[k].reshape(-1)) for k in
                     ['global_orient', 'body_pose', 'left_hand_pose']}

        k = 'right_hand_pose'
        body_loss[k] = .3*self.LossL2(rotmat2aa(self.sbj_params[k]).detach().reshape(-1), self.opt_params[k].reshape(-1))
        body_loss['transl'] = self.LossL1(self.opt_params['transl'],self.sbj_params['transl'].detach())

        losses.update(body_loss)

        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts, output

    def get_peneteration(self,source_mesh, target_mesh):

        source_verts = source_mesh.verts_packed()
        source_normals = source_mesh.verts_normals_packed()

        target_verts = target_mesh.verts_packed()
        target_normals = target_mesh.verts_normals_packed()

        src2trgt, trgt2src, src2trgt_idx, trgt2src_idx = self.ch_dist(source_verts.reshape(1,-1,3).to(self.device), target_verts.reshape(1,-1,3).to(self.device))

        source2target_correspond = target_verts[src2trgt_idx.data.view(-1).long()]

        distance_vector = source_verts - source2target_correspond

        in_out = torch.bmm(source_normals.view(-1, 1, 3), distance_vector.view(-1, 3, 1)).view(-1).sign()

        src2trgt_signed = src2trgt * in_out

        return src2trgt_signed


    def fitting(self, batch, net_output):

        cnet_verts, cnet_s_verts = self.get_smplx_verts(batch, net_output)

        for stg, optimizer in enumerate(self.optimizers):
            for itr in range(self.num_iters[stg]):
                optimizer.zero_grad()
                losses, opt_verts, opt_output = self.calc_loss(batch, net_output, stg)
                losses['loss_total'].backward(retain_graph=True)
                optimizer.step()
                if self.verbose and itr % 50 == 0:
                    print(self.create_loss_message(losses, stg, itr))

        opt_results = {k:aa2rotmat(v.detach()) for k,v in self.opt_params.items() if v != 'transl'}
        opt_results['transl'] = self.opt_params['transl'].detach()
        opt_results['fullpose_rotmat'] = opt_output.full_pose.detach()

        opt_results['cnet_verts'] = cnet_verts
        opt_results['opt_verts'] = opt_verts

        return opt_results

    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'


if __name__ == "__main__":
    
    pass