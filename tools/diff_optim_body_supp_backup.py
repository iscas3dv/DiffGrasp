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

from smplx import SMPLXLayer 
from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from bps_torch.bps import bps_torch
import chamfer_distance as chd
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces
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
        
        model_path = os.path.join('para_models/smplx')
                    
        self.female_model = SMPLXLayer(
                            model_path=model_path,
                            gender='female',
                            num_pca_comps=45,
                            flat_hand_mean=True,
                        ).to(self.device)
        self.male_model = SMPLXLayer(
                            model_path=model_path,
                            gender='male',
                            num_pca_comps=45,
                            flat_hand_mean=True,
                        ).to(self.device)
        self.ds_info = ds_info
        self.smplx_left_mask = torch.load(os.path.join(self.cfg.datasets.dataset_dir, 'obj_mesh', 'smplx_left_mask_bool_list.pt'))
        self.smplx_right_mask = torch.load(os.path.join(self.cfg.datasets.dataset_dir, 'obj_mesh', 'smplx_right_mask_bool_list.pt'))
        self.mano_bool_list = torch.from_numpy(np.load('MANO_bool_list.npy')).to(torch.bool)
        correspondence_M_S_idx = 'MANO_SMPLX_vertex_ids.pkl'
        self.bool_list_l = np.zeros([10475])
        self.bool_list_r = np.zeros([10475])
        self.bool_list_l_small = np.zeros([10475])
        self.bool_list_r_small = np.zeros([10475])

        with open(correspondence_M_S_idx, 'rb') as f:
            import pickle
            idxs_data = pickle.load(f)
            self.bool_list_l[idxs_data['left_hand']] = 1
            self.bool_list_r[idxs_data['right_hand']] = 1
            self.bool_list_l_small[idxs_data['left_hand'][self.mano_bool_list]] = 1
            self.bool_list_r_small[idxs_data['right_hand'][self.mano_bool_list]] = 1
        
        self.smplx_mano_left_verts_bool = torch.from_numpy(self.bool_list_l).to(torch.bool)
        self.smplx_mano_right_verts_bool = torch.from_numpy(self.bool_list_r).to(torch.bool)
        self.smplx_mano_left_verts_small_bool = torch.from_numpy(self.bool_list_l_small).to(torch.bool)
        self.smplx_mano_right_verts_small_bool = torch.from_numpy(self.bool_list_r_small).to(torch.bool)
        
    def recalcu_wrist_joints(self, batch, lr2omass):
        ### first frame rt
        lr2omass = lr2omass.reshape(30,2,3)
        omass = batch['obj_mass'].reshape(30,3).to(self.device)
        obj_transl = batch['obj_transl'].reshape(30,3).to(self.device)
        # obj_rotmat = batch['obj_orient_rotmat'].reshape(30, 3, 3).to(self.device)
        # Pomass_world = torch.matmul(omass[0] - obj_transl[0], obj_rotmat[0])
        joints_lr_position = lr2omass + omass.unsqueeze(1).repeat(1,2,1)
        
        # sp = Mesh(filename='/home/guest/hq/hoi/sphere.obj')
        # bsp_l_v, bsp_r_v, blr_colors = self.vis_lr_joints(joints_lr_position, sp.v, batch['contact_lable'][0])
        
        r = batch['obj_orient_rotmat'].reshape(30,3,3).to(self.device)
        for i in range(30):
            if self.sub_contact_frames[i][0]:
                
                parent_id_l = self.parent_l[i]
                Plj_world = torch.matmul((omass[parent_id_l] + lr2omass[parent_id_l][0]  - obj_transl[parent_id_l]),r[parent_id_l])
                joints_lr_position[i][0] = torch.matmul(Plj_world, r[i].permute(1,0)) + obj_transl[i]
                
            if self.sub_contact_frames[i][1]:
                # parent_id_r = self.parent_r[i]
                # p = lr2omass[parent_id_r][1]
                # H_i = omass[i] + torch.matmul(p ,torch.matmul(r[i].permute(1,0), r[parent_id_r]))
                # joints_lr_position[i][1] = H_i
                
                parent_id_r = self.parent_r[i]
                Plj_world = torch.matmul((omass[parent_id_r] + lr2omass[parent_id_r][1]  - obj_transl[parent_id_r]),r[parent_id_r])
                joints_lr_position[i][1] = torch.matmul(Plj_world, r[i].permute(1,0)) + obj_transl[i]

        self.new_joints_lr_position = joints_lr_position
        # asp_l_v, asp_r_v, alr_colors = self.vis_lr_joints(joints_lr_position, sp.v, batch['contact_lable'][0])
        
        # sp_anim = sp_animation()
        # for frame_idx in range(30):

        #     mesh_obj = Mesh(v=to_cpu(self.obj_vertice_seqs[frame_idx]),f=self.obj_face,vc = name_to_rgb['yellow'])
        #     mesh_obj.write_obj('/home/guest/hq/hoi/seqs_refine_mesh/mesh_obj_'+str(frame_idx)+'.obj')

        #     # mesh_gt = Mesh(v = to_cpu(recon_results['pred_verts'][frame_idx]), f = self.female_model.faces , vc=name_to_rgb['pink'])
        #     bmesh_lj = Mesh(v = to_cpu(bsp_l_v[frame_idx]), f = sp.f, vc=name_to_rgb[blr_colors[frame_idx][0]])
        #     bmesh_lj.write_obj('/home/guest/hq/hoi/seqs_refine_mesh/sphere_lj_'+str(frame_idx)+'.obj')
        #     bmesh_rj = Mesh(v = to_cpu(bsp_r_v[frame_idx]), f = sp.f, vc=name_to_rgb[blr_colors[frame_idx][1]])
        #     bmesh_rj.write_obj('/home/guest/hq/hoi/seqs_refine_mesh/sphere_rj_'+str(frame_idx)+'.obj')
        #     amesh_lj = Mesh(v = to_cpu(asp_l_v[frame_idx]), f = sp.f, vc=name_to_rgb[alr_colors[frame_idx][0]])
        #     amesh_lj.write_obj('/home/guest/hq/hoi/seqs_refine_mesh/sphere_lj_refine_'+str(frame_idx)+'.obj')
        #     amesh_rj = Mesh(v = to_cpu(asp_r_v[frame_idx]), f = sp.f, vc=name_to_rgb[alr_colors[frame_idx][1]])
        #     amesh_rj.write_obj('/home/guest/hq/hoi/seqs_refine_mesh/sphere_rj_refine_'+str(frame_idx)+'.obj')
        #     sp_anim.add_frame([mesh_obj,bmesh_lj,bmesh_rj,amesh_lj,amesh_rj], 
        #                         ['mesh_obj', 'bmesh_lj','bmesh_rj','amesh_lj','amesh_rj'])
        # animation_path = os.path.join('/home/guest/hq/hoi_vis/seq_path','refine')
        # if not os.path.exists(animation_path): os.makedirs(animation_path)
        # sp_anim.save_animation(os.path.join(animation_path, 'refine_'+str(self.it)+'.html'))
        return joints_lr_position
        # exit()
        
    def recon_obj_meshes(self, batch, idx2obj, it):
        self.it = it
        def get_object_vertices(arti, rot_mat, transl, part_label, verts_tem):
            bs = arti.shape[0]
            
            import pytorch3d.transforms.rotation_conversions as p3d_trans
            arti_mat = p3d_trans._axis_angle_rotation('Z', -arti)

            verts_tem = torch.from_numpy(verts_tem) # convert to meter.

            # rotate around the articulation
            index_bottom = np.where(part_label == 0)[0]
            rot_tem = arti_mat.to(torch.float32)
            verts_tem = verts_tem.repeat(bs, 1, 1).to(torch.float32)
            
            verts_tem[:,index_bottom] = torch.matmul(verts_tem[:,index_bottom], rot_tem.permute(0,2,1))

            verts_o = torch.matmul(verts_tem, rot_mat.permute(0,2,1)) + transl.unsqueeze(1)

            return verts_o
        
        seq_idx = 0
        obj_name = idx2obj[batch['obj_idx'][seq_idx,0].item()]
        obj_path = os.path.join(self.cfg.datasets.dataset_dir, 'obj_mesh', obj_name+'.obj')
        obj_template = Mesh(filename=obj_path)

        obj_verts = torch.from_numpy(obj_template.v).to(torch.float32)
        obj_verts = obj_verts.repeat(30,1,1)
        obj_vertice = torch.matmul(obj_verts, batch['obj_orient_rotmat'][seq_idx,...]) + batch['obj_transl'][seq_idx,...].unsqueeze(1)
                
        
        # part_label = np.load(os.path.join(self.cfg.datasets.dataset_dir, 'obj_mesh', '%s_label.npy'%obj_name))
        # obj_vertice = get_object_vertices(arti = batch['obj_arti'][seq_idx,...],
        #                                     rot_mat = batch['obj_orient_rotmat'][seq_idx,...],
        #                                     transl = batch['obj_transl'][seq_idx,...], 
        #                                     part_label = part_label,
        #                                     verts_tem = obj_template.v)
        
        self.obj_vertice_seqs = obj_vertice
        self.obj_face = obj_template.f

        first_contact_frames, sub_contact_frames, parent_l, parent_r = self.get_key_contact_frame(batch['contact_lable'])
        self.first_contact_frames = first_contact_frames
        self.sub_contact_frames = sub_contact_frames
        self.parent_l = parent_l
        self.parent_r = parent_r
        
        
    def prepare_recon(self, batch, sampled_results):
        
        recon_output = {}

        pred_poses = torch.cat((sampled_results['smplx_rot2world'].unsqueeze(1), sampled_results['pose_rotmat']),dim=1)
        pred_results = self.smplxLayer(batch, poses=pred_poses, trans=sampled_results['smplx_transl2world'], is_6D=True)
        recon_output['pred_verts'] = pred_results['verts']
        recon_output['pred_joints'] = pred_results['joints']
        
        recon_output['sampled_results'] = sampled_results
        
        return recon_output
    
    def smplxLayer(self, batch, poses, trans, is_6D):


        fs = poses.shape[0]
        poses = poses.reshape(fs,55,-1,3).to(self.device)
        trans = trans.reshape(fs,3).to(self.device)
        bparams = parms_6D2full(poses, trans, d62rot=is_6D)
        
        genders = batch['gender'].reshape(-1)
        males = genders == 1
        females = ~males
        
        v_template = batch['smplx_v_temp'].reshape(-1,10475,3).to(self.device)
        
        FN = sum(females)
        MN = sum(males)
        
        recon_output = {}
        recon_output['verts'] = torch.zeros(fs,10475,3).to(self.device)
        recon_output['joints'] = torch.zeros(fs,127,3).to(self.device)
        if FN > 0:

            f_params = {k: v[females] for k, v in bparams.items()}
            self.female_model.v_template = v_template[females].clone().to(torch.float32)
            f_output = self.female_model(**f_params)
            recon_output['verts'][females] = f_output.vertices
            recon_output['joints'][females] = f_output.joints
        
        if MN > 0:

            m_params = {k: v[males] for k, v in bparams.items()}
            self.male_model.v_template = v_template[males].clone().to(torch.float32)
            m_output = self.male_model(**m_params)
            recon_output['verts'][males] = m_output.vertices
            recon_output['joints'][males] = m_output.joints
            # m_verts = m_output.vertices
            # m_joints = m_output.joints
            
            # recon_output['m_verts'] = m_verts
            # recon_output['m_joints'] = m_joints
            # recon_output['m_params'] = m_params
            
        return recon_output
    def load_opt_result(self):
        return self.latest_opt_results
    def fitting(self, batch, net_output, t):
        with torch.enable_grad():
            net_output = net_output.squeeze(1).permute(0,2,1)
            # torch.Size([bs, 30, 198])
            
            net_outdict = {'smplx_transl2world': net_output[..., 0:3], 
                   'pose_rotmat': net_output[..., 3:327],
                   'smplx_rot2world':net_output[...,327:333],
                   'lr2omass': net_output[...,333:339],
                   }
            
            # net_outdict = {'transl2obj': net_output[..., :3], 
            #        'smplx_transl2world': net_output[..., 3:6], 
            #        'rot2obj_rotmat': net_output[..., 6:12], 
            #        'pose_rotmat': net_output[..., 12:336],
            #        'smplx_rot2world':net_output[...,336:342],
            #        'lr_transl': net_output[...,342:348],
            #        'lr2omass': net_output[...,348:354],
            #        }
            
            refine_out_dict = {k:v.detach().clone() for k,v in net_outdict.items()}
            
            norm_pose = self.ds_info['smplx']
            # 先反归一化出来
            net_outdict = self.dict_destandardization(net_outdict, norm_pose)
            
            before_verts = self.prepare_recon(batch, net_outdict)['pred_verts']
            smplx_face = self.female_model.faces
            
            ### ground contact refinement stage
            lr_grd_contact = 1e-3
            opt_params_gnd = {'smplx_transl2world': nn.Parameter(net_outdict['smplx_transl2world'].reshape(30,3)), 
                               'smplx_rot2world': nn.Parameter(net_outdict['smplx_rot2world'].reshape(30,2,3)), 
                               'pose_rotmat': nn.Parameter(net_outdict['pose_rotmat'].reshape(30,54,2,3))}
            
            opt_grd_contact = optim.Adam([opt_params_gnd[k] for k in ['smplx_rot2world', 'smplx_transl2world']], lr=lr_grd_contact)
            num_iters_grn = 100
            for itr in range(num_iters_grn):
                opt_grd_contact.zero_grad()
                recon_results = self.prepare_recon(batch, opt_params_gnd)
                loss_total, losses = self.calc_grd_contact_loss(batch, recon_results)
                loss_total.backward()
                opt_grd_contact.step()
                if itr % 10 == 0:
                    print(self.create_loss_message(losses, itr))
                    
            for k, v in opt_params_gnd.items():
                if k in ['smplx_transl2world','smplx_rot2world']:
                    net_outdict[k] = v.detach().clone()
            
            ### lr wrist joints stable refinement stage
            lr_joints_consis = 4e-3
            lr_lr2omass_refine = 1e-4

            upper_body_rotmat = nn.Parameter(net_outdict['pose_rotmat'][:,12:21,...])
            
            opt_params_joints = {'smplx_transl2world': net_outdict['smplx_transl2world'].reshape(30,3), 
                               'smplx_rot2world': net_outdict['smplx_rot2world'].reshape(30,2,3), 
                               'pose_rotmat': nn.Parameter(net_outdict['pose_rotmat']),
                               'lr2omass': nn.Parameter(net_outdict['lr2omass'].reshape(30,2,3))}
            
            # opt_joints_consis = optim.Adam([upper_body_rotmat], lr=lr_joints_consis)
            opt_joints_consis = optim.Adam([upper_body_rotmat], lr=lr_joints_consis)
            opt_lr2omass_refine = optim.Adam([opt_params_joints['lr2omass']], lr=lr_lr2omass_refine)

            num_iters_joints = 100
            for itr in range(num_iters_joints):
                opt_joints_consis.zero_grad()
                opt_lr2omass_refine.zero_grad()
                
                recon_results = self.prepare_recon(batch, opt_params_joints)
                loss_total, losses = self.calc_joints_consis_loss(batch, recon_results, opt_params_joints['lr2omass'])

                loss_total.backward()

                upper_body_rotmat.grad = opt_params_joints['pose_rotmat'].grad[:,12:21,...]

                opt_joints_consis.step()
                opt_lr2omass_refine.step()
                
                if itr % 10 == 0:
                    print(self.create_loss_message(losses, itr))
                    
            net_outdict['pose_rotmat'][:,12:21,...] = upper_body_rotmat.detach().clone()
            net_outdict['lr2omass'] = opt_params_joints['lr2omass'].detach().clone()
            
            ### lr contact and peneration refinement stage ###
            lr_hand_contact = 1e-3
            num_iters_contact = 100
            #### left hand
            left_hand_idx = [i+24 for i in range(15)]
            left_upper_body_idx = [15,17,19]
            left_upper_body_idx.extend(left_hand_idx) 
            # [15, 17, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
            l_hands_rotmat = [nn.Parameter(net_outdict['pose_rotmat'][:,15:16,...]),
                                nn.Parameter(net_outdict['pose_rotmat'][:,17:18,...]),
                                nn.Parameter(net_outdict['pose_rotmat'][:,19:20,...]),
                                nn.Parameter(net_outdict['pose_rotmat'][:,24:39,...])]
            
            opt_params_contact = {'smplx_transl2world': net_outdict['smplx_transl2world'].reshape(30,3), 
                            'smplx_rot2world': net_outdict['smplx_rot2world'].reshape(30,2,3), 
                            'pose_rotmat': nn.Parameter(net_outdict['pose_rotmat'])}
            
            opt_lhand_obj_contact = optim.Adam(l_hands_rotmat, lr=lr_hand_contact)
            
            for itr in range(num_iters_contact):
                opt_lhand_obj_contact.zero_grad()
                recon_results = self.prepare_recon(batch, opt_params_contact)
                loss_total, losses = self.calc_hand_obj_contact_loss(batch, recon_results, is_left = True)
                if loss_total == 0:
                    print('[no contact]' + self.create_loss_message(losses, itr))
                    break
                loss_total.backward()
                l_hands_rotmat[0].grad = opt_params_contact['pose_rotmat'].grad[:,15:16,...]
                l_hands_rotmat[1].grad = opt_params_contact['pose_rotmat'].grad[:,17:18,...]
                l_hands_rotmat[2].grad = opt_params_contact['pose_rotmat'].grad[:,19:20,...]
                l_hands_rotmat[3].grad = opt_params_contact['pose_rotmat'].grad[:,24:39,...]
                opt_lhand_obj_contact.step()
                if itr % 1 == 0:
                    print(self.create_loss_message(losses, itr))
                    
                net_outdict['pose_rotmat'][:,15:16,...] = l_hands_rotmat[0].detach().clone()
                net_outdict['pose_rotmat'][:,17:18,...] = l_hands_rotmat[1].detach().clone()
                net_outdict['pose_rotmat'][:,19:20,...] = l_hands_rotmat[2].detach().clone()
                net_outdict['pose_rotmat'][:,24:39,...] = l_hands_rotmat[3].detach().clone()
                
            #### right hand
            right_hand_idx = [i+39 for i in range(15)]
            right_upper_body_idx = [16,18,20]
            right_upper_body_idx.extend(right_hand_idx)
            # [16, 18, 20, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
            r_hands_rotmat = [nn.Parameter(net_outdict['pose_rotmat'][:,16:17,...]),
                                nn.Parameter(net_outdict['pose_rotmat'][:,18:19,...]),
                                nn.Parameter(net_outdict['pose_rotmat'][:,20:21,...]),
                                nn.Parameter(net_outdict['pose_rotmat'][:,39:54,...])]
            opt_params_contact = {'smplx_transl2world': net_outdict['smplx_transl2world'].reshape(30,3), 
                            'smplx_rot2world': net_outdict['smplx_rot2world'].reshape(30,2,3), 
                            'pose_rotmat': nn.Parameter(net_outdict['pose_rotmat'])}
            opt_rhand_obj_contact = optim.Adam(r_hands_rotmat, lr=lr_hand_contact)
            
            for itr in range(num_iters_contact):
                opt_rhand_obj_contact.zero_grad()
                recon_results = self.prepare_recon(batch, opt_params_contact)
                loss_total, losses = self.calc_hand_obj_contact_loss(batch, recon_results,is_left = False)
                if loss_total == 0:
                    print('[no contact]' + self.create_loss_message(losses, itr))
                    break
                loss_total.backward()
                r_hands_rotmat[0].grad = opt_params_contact['pose_rotmat'].grad[:,16:17,...]
                r_hands_rotmat[1].grad = opt_params_contact['pose_rotmat'].grad[:,18:19,...]
                r_hands_rotmat[2].grad = opt_params_contact['pose_rotmat'].grad[:,20:21,...]
                r_hands_rotmat[3].grad = opt_params_contact['pose_rotmat'].grad[:,39:54,...]
                opt_rhand_obj_contact.step()
                if itr % 1 == 0:
                    print(self.create_loss_message(losses, itr))
            
            net_outdict['pose_rotmat'][:,16:17,...] = r_hands_rotmat[0].detach().clone()
            net_outdict['pose_rotmat'][:,18:19,...] = r_hands_rotmat[1].detach().clone()
            net_outdict['pose_rotmat'][:,20:21,...] = r_hands_rotmat[2].detach().clone()
            net_outdict['pose_rotmat'][:,39:54,...] = r_hands_rotmat[3].detach().clone()
                
                
            
            # after_verts = self.prepare_recon(batch, net_outdict)['pred_verts']
            # sp_anim = sp_animation()
            # for i in range(30):
            #     mesh_obj = Mesh(v=to_cpu(self.obj_vertice_seqs[i]),f=self.obj_face,vc = name_to_rgb['yellow'])
            #     after_mesh = Mesh(v=to_cpu(after_verts[i]), f=smplx_face, vc=name_to_rgb['red'])
            #     before_mesh = Mesh(v=to_cpu(before_verts[i]), f=smplx_face, vc=name_to_rgb['green'])
            #     sp_anim.add_frame([mesh_obj, after_mesh, before_mesh], 
            #                       ['mesh_obj', 'after_mesh', 'before_mesh'])               
            # animation_path = os.path.join('/home/guest/hq/hoi_vis/seq_path','refine')
            # if not os.path.exists(animation_path): os.makedirs(animation_path)
            # sp_anim.save_animation(os.path.join(animation_path, 'refine_'+str(self.it)+'.html'))
            
            # exit()
            
            refine_out_dict = self.dict_standardization(net_outdict, norm_pose)
            
            opt_results = {}
            opt_results['img'] = torch.cat([v.detach().clone().reshape(self.bs, self.fs,-1) for v in refine_out_dict.values()], dim=2).permute(0,2,1).unsqueeze(1)
            self.latest_opt_results = opt_results
            return opt_results
    
    
    def vis_lr_joints(self, lr_joints, basis_points, contact_label):
        
        sp_l_transl = lr_joints[:,0,:].detach().cpu()
        sp_r_transl = lr_joints[:,1,:].detach().cpu()
        
        sp_l_v = torch.tensor(basis_points).reshape(1,-1,3).repeat(30,1,1) + sp_l_transl.reshape(30,1,3)
        sp_r_v = torch.tensor(basis_points).reshape(1,-1,3).repeat(30,1,1) + sp_r_transl.reshape(30,1,3)
        
        contact_label_dict = [['blue','green'],
                              ['blue','red'],
                              ['red','green'],
                              ['red','red']]
        
        contact_counts = contact_label[:,0] * 2 +  contact_label[:,1]
        
        color_seqs = []
        for i in range(30):
            color_seqs.append(contact_label_dict[contact_counts[i]])
        
        return sp_l_v, sp_r_v, color_seqs
    
    def calc_joints_consis_loss(self, batch, recon_results, lr2omass):
        
        new_joints_lr_position = self.recalcu_wrist_joints(batch, lr2omass.detach().clone()).detach().clone()

        losses = {}
        contact_label = batch['contact_lable'].reshape(-1,2).to(self.device)
        
        ## masked_wrist_joints_loss
        joints_loss_weight = 1
        losses['wrist_joints'] = 0
        # print(recon_results['pred_joints'].grad)
        dist_lr = torch.norm(new_joints_lr_position - recon_results['pred_joints'][:,20:22,:]) # 30,2
        
        losses['wrist_joints'] += torch.mean(contact_label * dist_lr)
        losses['wrist_joints'] *= joints_loss_weight 
        

        ## masked lr2omass loss
        lr2omass_loss_weight = 1
        losses['lr2omass_loss'] = 0
        dist_lr2omass = torch.norm(new_joints_lr_position.reshape(30,2,3)
                                   - (batch['obj_mass'].reshape(30,1,3).to(self.device) + lr2omass))
        losses['lr2omass_loss'] += torch.mean(contact_label * dist_lr2omass)
        losses['lr2omass_loss'] *= lr2omass_loss_weight

        loss_total = torch.stack(list(losses.values())).sum()
        
        losses['loss_total'] = loss_total
        
        return loss_total, losses
    
    def calc_hand_obj_contact_loss(self, batch, recon_results, is_left):
        
        losses = {}
        contact_label = batch['contact_lable'].reshape(-1,2).to(self.device)
        ## pene_loss
        p_c_loss_weight = 10
        pene_loss_weight = 0.5
        contact_loss_weight = 1 - pene_loss_weight
        losses['penetration_loss'] = 0
        ## contact_loss 
        losses['contact_loss'] = 0
        
        verts_device = recon_results['pred_verts'].device
        obj_face = torch.from_numpy(self.obj_face.astype(np.int64))
        
        object_template_mesh = index_vertices_by_faces(self.obj_vertice_seqs, 
                                                       obj_face.to(self.obj_vertice_seqs.device))
        if is_left:
        # torch.Size([30, 10475])
            lh_sampled_verts = recon_results['pred_verts'][:,self.smplx_mano_left_verts_small_bool]
            
            inside = kaolin.ops.mesh.check_sign(self.obj_vertice_seqs.to(verts_device), obj_face.to(verts_device), lh_sampled_verts)
            distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(lh_sampled_verts.contiguous(),
                                                                                object_template_mesh.to(verts_device))
            if inside.any():
                losses['penetration_loss'] += distance[inside].sum()
                losses['penetration_loss'] *= pene_loss_weight * p_c_loss_weight
            else:
                losses['penetration_loss'] = torch.tensor(0., dtype=torch.float32).to(lh_sampled_verts.device)
            # print(distance.shape)
            left_masked_contact_loss = torch.mean(torch.abs(distance * contact_label[:,0:1]))
            
            losses['contact_loss'] += left_masked_contact_loss # need for splits
            losses['contact_loss'] *= contact_loss_weight * p_c_loss_weight
        else:
            rh_sampled_verts = recon_results['pred_verts'][:,self.smplx_mano_right_verts_small_bool]
            
            inside = kaolin.ops.mesh.check_sign(self.obj_vertice_seqs.to(verts_device), obj_face.to(verts_device), rh_sampled_verts)
            distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(rh_sampled_verts.contiguous(),
                                                                                object_template_mesh.to(verts_device))
            if inside.any():
                losses['penetration_loss'] += distance[inside].sum()
                losses['penetration_loss'] *= pene_loss_weight * p_c_loss_weight
            else:
                losses['penetration_loss'] = torch.tensor(0., dtype=torch.float32).to(rh_sampled_verts.device)
            # print(distance.shape)  
            right_masked_contact_loss = torch.mean(torch.abs(distance * contact_label[:,1:2]))
            
            losses['contact_loss'] += right_masked_contact_loss # need for splits
            losses['contact_loss'] *= contact_loss_weight * p_c_loss_weight
            
        loss_total = torch.stack(list(losses.values())).sum()
        
        losses['loss_total'] = loss_total
        
        return loss_total, losses
        
        
        # asp_l_v, asp_r_v, alr_colors = self.vis_lr_joints(joints_lr_position, sp.v, batch['contact_lable'][0])
        
        # sp_anim = sp_animation()
        # for frame_idx in range(30):

        #     mesh_obj = Mesh(v=to_cpu(self.obj_vertice_seqs[frame_idx]),f=self.obj_face,vc = name_to_rgb['yellow'])

        #     mesh_gt = Mesh(v = to_cpu(recon_results['pred_verts'][frame_idx]), f = self.female_model.faces , vc=name_to_rgb['pink'])
        #     bmesh_lj = Mesh(v = to_cpu(bsp_l_v[frame_idx]), f = sp.f, vc=name_to_rgb[blr_colors[frame_idx][0]])
        #     bmesh_rj = Mesh(v = to_cpu(bsp_r_v[frame_idx]), f = sp.f, vc=name_to_rgb[blr_colors[frame_idx][1]])
        #     amesh_lj = Mesh(v = to_cpu(asp_l_v[frame_idx]), f = sp.f, vc=name_to_rgb[alr_colors[frame_idx][0]])
        #     amesh_rj = Mesh(v = to_cpu(asp_r_v[frame_idx]), f = sp.f, vc=name_to_rgb[alr_colors[frame_idx][1]])
        #     sp_anim.add_frame([mesh_obj, mesh_gt,bmesh_lj,bmesh_rj,amesh_lj,amesh_rj], 
        #                         ['mesh_obj', 'mesh_gt', 'bmesh_lj','bmesh_rj','amesh_lj','amesh_rj'])
        # animation_path = os.path.join('/home/guest/hq/hoi_vis/seq_path','refine')
        # if not os.path.exists(animation_path): os.makedirs(animation_path)
        # sp_anim.save_animation(os.path.join(animation_path, 'refine.html'))
        # exit()
        pass
    
    def calc_grd_contact_loss(self, batch, recon_results):
        
        losses = {}
        grnd_loss_weight = 1
        first_frame_z = recon_results['pred_joints'][0:1,60:66,2].detach().clone().repeat(30,1)
        # foot_z = torch.mean(torch.abs(recon_results['pred_joints'][...,60:66,2]))
        losses['grnd_loss'] = 0
        losses['grnd_loss'] += self.LossL2(first_frame_z, recon_results['pred_joints'][...,60:66,2])
        losses['grnd_loss'] *= grnd_loss_weight
        
        loss_total = torch.stack(list(losses.values())).sum()
        
        losses['loss_total'] = loss_total
        
        return loss_total, losses
    
    def get_key_contact_frame(self, contact_label):
        contact_label = contact_label[0]
        
        first_contact_frames = torch.zeros_like(contact_label)
        sub_contact_frames = torch.zeros_like(contact_label)
        contact_label_l = contact_label.permute(1,0)[0]
        contact_label_r = contact_label.permute(1,0)[1]
        # print(contact_label_l)
        # print(contact_label_r)
        # first frame contact detect
        first_contact_frames[0] = contact_label[0]
            # check others frames contact
                
        for i in range(29):
            i = i + 1
            if contact_label_l[i-1] == 0 and contact_label_l[i] == 1:
                first_contact_frames[i][0] = 1
                
            if contact_label_r[i-1] == 0 and contact_label_r[i] == 1:
                first_contact_frames[i][1] = 1
        # sub_contact frame detect
        contact_label_bool = contact_label.to(torch.bool)
        first_contact_frames_bool = first_contact_frames.to(torch.bool)
        sub_contact_frames_bool = torch.logical_xor(first_contact_frames_bool, contact_label_bool)

        parent_l = [-1] * 30
        parent_r = [-1] * 30
        
        start_index_l = None
        start_index_r = None

        for i in range(30):
            if contact_label_l[i] == 1:
                if start_index_l is None:
                    start_index_l = i
                    # output_sequence[i] = i
                else:
                    parent_l[i] = start_index_l
            else:
                start_index_l = None
            if contact_label_r[i] == 1:
                if start_index_r is None:
                    start_index_r = i
                    # output_sequence[i] = i
                else:
                    parent_r[i] = start_index_r
            else:
                start_index_r = None

        return first_contact_frames_bool, sub_contact_frames_bool, parent_l, parent_r
        
    
    def calc_loss(self, batch):
        # ###### 注意！！目前bs只能为1
        
        fs = batch['obj_sdf_idx'].shape[1]
        ### 获得物体的mesh(目前所有操作都是在物体坐标系下)
        mesh_path = '/data/hq/arctic_single/obj_mesh'
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
        
        obj_norms_batch = torch.tensor(obj_fixed_mesh.estimate_vertex_normals().reshape(1,-1,3).repeat(fs, axis=0)).to(self.device).to(torch.float32)
        # obj_verts_batch = torch.tensor(obj_fixed_mesh.v.reshape(1,-1,3).repeat(fs*2, axis=0)).to(self.device).to(torch.float32)
        obj_verts_batch = obj_verts_for_vis.repeat(2,1,1)
        
        smplx_norms = []
        # lh_norms = []
        # rh_norms = []
        obj_norms = []

        recon_out_objcoords = self.prepare_recon(batch, self.opt_params, is_world=True)     
        if batch['gender'][0][0] == 1:
            face = self.male_model.faces
        else:
            face = self.female_model.faces
        sp_anim = sp_animation()
        for i in range(fs):
            obj_mesh = Mesh(v=to_cpu(obj_verts_for_vis)[i], f=obj_fixed_mesh.f,vc=name_to_rgb['yellow'])
            smplx_mesh = Mesh(v=to_cpu(recon_out_objcoords['pred_verts'][0][i]), f=face, vc=name_to_rgb['green'])

            sp_anim.add_frame([obj_mesh, smplx_mesh], ['obj_mesh', 'smplx_mesh'])
            obj_norms.append(obj_mesh.estimate_vertex_normals())
            smplx_norms.append(smplx_mesh.estimate_vertex_normals())

            
        # sp_anim.save_animation('./vis.html')
        obj_norms = torch.from_numpy(np.array(obj_norms))
        obj_norms = obj_norms.repeat(2,1,1).to(self.device).to(torch.float32)
    
        smplx_norms = torch.from_numpy(np.array(smplx_norms)).to(self.device).to(torch.float32)
        # lh_norms = torch.from_numpy(np.array(lh_norms))
        # rh_norms = torch.from_numpy(np.array(rh_norms))
        
        # lr_norms = torch.concatenate((lh_norms,rh_norms),dim=0).to(self.device).to(torch.float32)
        
        # lr_verts = torch.concatenate((recon_out_objcoords['lh_pred_vert'][0], recon_out_objcoords['rh_pred_vert'][0])).to(torch.float32)
        smplx_verts = recon_out_objcoords['pred_verts'][0].to(torch.float32)
        y2x_signed, x2y_signed, yidx_near, xidx_near = self.point2point_signed(obj_verts_batch, smplx_verts, obj_norms, smplx_norms)

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

    
    
    # def dict_destandardization(self, results, smplx_norm_pose, fs=30,):
        
    #     results['transl2obj'] = tensor_destandardization(results['transl2obj'], 
    #                                                   smplx_norm_pose['transl2obj_mean'].to(self.device), smplx_norm_pose['transl2obj_std'].to(self.device))
    #     results['smplx_transl2world'] = tensor_destandardization(results['smplx_transl2world'], 
    #                                                   smplx_norm_pose['transl2world_mean'].to(self.device), smplx_norm_pose['transl2world_std'].to(self.device))
    #     results['rot2obj_rotmat'] = tensor_destandardization(results['rot2obj_rotmat'].reshape(fs,2,3), 
    #                                                        smplx_norm_pose['rot2obj_mean'][:2,:].to(self.device), smplx_norm_pose['rot2obj_std'][:2,:].to(self.device))
    #     results['pose_rotmat'] = tensor_destandardization(results['pose_rotmat'].reshape(fs,-1,2,3), 
    #                                                        smplx_norm_pose['pose_mean'][:,:2,:].to(self.device), smplx_norm_pose['pose_std'][:,:2,:].to(self.device))
    #     results['smplx_rot2world']  = tensor_destandardization(results['smplx_rot2world'].reshape(fs,2,3), 
    #                                                        smplx_norm_pose['rot2world_mean'][:2,:].to(self.device), smplx_norm_pose['rot2world_std'][:2,:].to(self.device))
    #     results['lr_transl'] = tensor_destandardization(results['lr_transl'].reshape(fs,2,3),
    #                                             smplx_norm_pose['joints_mean'][20:22,:].to(self.device), smplx_norm_pose['joints_std'][20:22,:].to(self.device))
    #     results['lr2omass'] = tensor_destandardization(results['lr2omass'].reshape(fs,2,3), 
    #                                                   smplx_norm_pose['lr_2omass_mean'].to(self.device), smplx_norm_pose['lr_2omass_std'].to(self.device))
    #     return results
    
    
    def dict_destandardization(self, results, smplx_norm_pose, fs=30,):
        
        
        results['smplx_transl2world'] = tensor_destandardization(results['smplx_transl2world'], 
                                                      smplx_norm_pose['transl2world_mean'].to(self.device), smplx_norm_pose['transl2world_std'].to(self.device))
        
        results['pose_rotmat'] = tensor_destandardization(results['pose_rotmat'].reshape(fs,-1,2,3), 
                                                           smplx_norm_pose['pose_mean'][:,:2,:].to(self.device), smplx_norm_pose['pose_std'][:,:2,:].to(self.device))
        results['smplx_rot2world']  = tensor_destandardization(results['smplx_rot2world'].reshape(fs,2,3), 
                                                           smplx_norm_pose['rot2world_mean'][:2,:].to(self.device), smplx_norm_pose['rot2world_std'][:2,:].to(self.device))
       
        results['lr2omass'] = tensor_destandardization(results['lr2omass'].reshape(fs,2,3), 
                                                      smplx_norm_pose['lr_2omass_mean'].to(self.device), smplx_norm_pose['lr_2omass_std'].to(self.device))
        return results
    
    
    
    # def dict_standardization(self, results, smplx_norm_pose, fs=30):
        
    #     results['transl2obj'] = tensor_standardization(results['transl2obj'], 
    #                                                   smplx_norm_pose['transl2obj_mean'].to(self.device), smplx_norm_pose['transl2obj_std'].to(self.device)).reshape(30,-1)
    #     results['smplx_transl2world'] = tensor_standardization(results['smplx_transl2world'], 
    #                                                   smplx_norm_pose['transl2world_mean'].to(self.device), smplx_norm_pose['transl2world_std'].to(self.device)).reshape(30,-1)
    #     results['rot2obj_rotmat'] = tensor_standardization(results['rot2obj_rotmat'].reshape(fs,2,3), 
    #                                                        smplx_norm_pose['rot2obj_mean'][:2,:].to(self.device), smplx_norm_pose['rot2obj_std'][:2,:].to(self.device)).reshape(30,-1)
    #     results['pose_rotmat'] = tensor_standardization(results['pose_rotmat'].reshape(fs,-1,2,3), 
    #                                                        smplx_norm_pose['pose_mean'][:,:2,:].to(self.device), smplx_norm_pose['pose_std'][:,:2,:].to(self.device)).reshape(30,-1)
    #     results['smplx_rot2world']  = tensor_standardization(results['smplx_rot2world'].reshape(fs,2,3), 
    #                                                        smplx_norm_pose['rot2world_mean'][:2,:].to(self.device), smplx_norm_pose['rot2world_std'][:2,:].to(self.device)).reshape(30,-1)
    #     results['lr_transl'] = tensor_standardization(results['lr_transl'].reshape(fs,2,3),
    #                                             smplx_norm_pose['joints_mean'][20:22,:].to(self.device), smplx_norm_pose['joints_std'][20:22,:].to(self.device)).reshape(30,-1)
    #     results['lr2omass'] = tensor_standardization(results['lr2omass'].reshape(fs,2,3), 
    #                                                   smplx_norm_pose['lr_2omass_mean'].to(self.device), smplx_norm_pose['lr_2omass_std'].to(self.device)).reshape(30,-1)
    #     return results
    
    def dict_standardization(self, results, smplx_norm_pose, fs=30):
        
        
        results['smplx_transl2world'] = tensor_standardization(results['smplx_transl2world'], 
                                                      smplx_norm_pose['transl2world_mean'].to(self.device), smplx_norm_pose['transl2world_std'].to(self.device)).reshape(30,-1)
        
        results['pose_rotmat'] = tensor_standardization(results['pose_rotmat'].reshape(fs,-1,2,3), 
                                                           smplx_norm_pose['pose_mean'][:,:2,:].to(self.device), smplx_norm_pose['pose_std'][:,:2,:].to(self.device)).reshape(30,-1)
        results['smplx_rot2world']  = tensor_standardization(results['smplx_rot2world'].reshape(fs,2,3), 
                                                           smplx_norm_pose['rot2world_mean'][:2,:].to(self.device), smplx_norm_pose['rot2world_std'][:2,:].to(self.device)).reshape(30,-1)
        
        results['lr2omass'] = tensor_standardization(results['lr2omass'].reshape(fs,2,3), 
                                                      smplx_norm_pose['lr_2omass_mean'].to(self.device), smplx_norm_pose['lr_2omass_std'].to(self.device)).reshape(30,-1)
        return results
    
    
    @staticmethod
    def create_loss_message(loss_dict, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'
    

if __name__ == "__main__":
    
    pass