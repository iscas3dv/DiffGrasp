import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os

from smplx import SMPLXLayer 
from models.model_utils import parms_6D2full

import kaolin
from kaolin.ops.mesh import index_vertices_by_faces

import trimesh
class DiffOptim(nn.Module):
    
    def __init__(self,
                 BS,
                 T,
                 obj_sdfs,
                 cfg=None):
        super(DiffOptim, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg

        self.BS = BS
        self.T = T
        self.obj_sdfs = obj_sdfs
        
        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')
        
        model_path = os.path.join('para_models/smplx')
                    
        self.female_model = SMPLXLayer(
                            model_path=model_path,
                            gender='female',
                            num_pca_comps=24,
                            flat_hand_mean=True,
                        ).to(self.device)
        self.male_model = SMPLXLayer(
                            model_path=model_path,
                            gender='male',
                            num_pca_comps=24,
                            flat_hand_mean=True,
                        ).to(self.device)
        
        self.mano_bool_list = torch.from_numpy(np.load('configs/MANO_bool_list.npy')).to(torch.bool)
        self.bool_list_l = np.zeros([10475])
        self.bool_list_r = np.zeros([10475])
        self.bool_list_l_small = np.zeros([10475])
        self.bool_list_r_small = np.zeros([10475])

        correspondence_M_S_idx = 'configs/MANO_SMPLX_vertex_ids.pkl'
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

        joints_lr_position = lr2omass + omass.unsqueeze(1).repeat(1,2,1)

        r = batch['obj_global_orient'].reshape(30,3,3).to(self.device)
        
        for i in range(30):
            if self.sub_contact_frames[i][0]:
                
                parent_id_l = self.parent_l[i]
                Plj_world = torch.matmul((omass[parent_id_l] + lr2omass[parent_id_l][0]  - obj_transl[parent_id_l]),r[parent_id_l])
                joints_lr_position[i][0] = torch.matmul(Plj_world, r[i].permute(1,0)) + obj_transl[i]
                
            if self.sub_contact_frames[i][1]:

                parent_id_r = self.parent_r[i]
                Prj_world = torch.matmul((omass[parent_id_r] + lr2omass[parent_id_r][1]  - obj_transl[parent_id_r]),r[parent_id_r].permute(1,0))
                joints_lr_position[i][1] = torch.matmul(Prj_world, r[i]) + obj_transl[i]

        self.new_joints_lr_position = joints_lr_position

        return joints_lr_position
        
    def recon_obj_meshes(self, batch, idx2obj, it):
        self.it = it
        
        seq_idx = 0
        obj_name = idx2obj[batch['obj_idx'][seq_idx,0].item()]
        obj_path = os.path.join(self.cfg.datasets.dataset_dir, 'obj_meshes', obj_name+'.obj')
        obj_template = trimesh.load(obj_path, process=False)

        obj_verts = torch.from_numpy(obj_template.vertices).to(torch.float32)
        obj_verts = obj_verts.repeat(30,1,1)
        obj_vertice = torch.matmul(obj_verts, batch['obj_global_orient'][seq_idx,...]) + batch['obj_transl'][seq_idx,...].unsqueeze(1)
        
        self.obj_vertice_seqs = obj_vertice
        self.obj_face = obj_template.faces

        first_contact_frames, sub_contact_frames, parent_l, parent_r = self.get_key_contact_frame(batch['contact_label'])
        self.first_contact_frames = first_contact_frames
        self.sub_contact_frames = sub_contact_frames
        self.parent_l = parent_l
        self.parent_r = parent_r
        
        
    def prepare_recon(self, batch, sampled_results):
        
        recon_output = {}

        pred_poses = torch.cat((sampled_results['sbj_global_orient'].unsqueeze(1), sampled_results['sbj_fullpose']), dim=1)
        pred_results = self.smplxLayer(batch, poses=pred_poses, trans=sampled_results['sbj_smplx_transl'], is_6D=True)
        recon_output['pred_verts'] = pred_results['verts']
        recon_output['pred_joints'] = pred_results['joints']
        
        recon_output['sampled_results'] = sampled_results
        
        return recon_output
    
    def smplxLayer(self, batch, poses, trans, is_6D):


        T = poses.shape[0]
        poses = poses.reshape(T,55,-1,3).to(self.device)
        trans = trans.reshape(T,3).to(self.device)
        bparams = parms_6D2full(poses, trans, d62rot=is_6D)
        
        genders = batch['gender'].reshape(-1)
        males = genders == 1
        females = ~males
        
        v_template = batch['smplx_v_temp'].reshape(-1,10475,3).to(self.device)
        
        FN = sum(females)
        MN = sum(males)
        
        recon_output = {}
        recon_output['verts'] = torch.zeros(T,10475,3).to(self.device)
        recon_output['joints'] = torch.zeros(T,127,3).to(self.device)
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

            
        return recon_output
    def load_opt_result(self):
        return self.latest_opt_results
    def fitting(self, batch, net_output, t):
        with torch.enable_grad():
            net_output = net_output.squeeze(1).permute(0,2,1)
            # torch.Size([BS, 30, 198])
            BS, T, _ = net_output.shape
            net_outdict = {
                'sbj_smplx_transl': net_output[..., 0:3].reshape(BS*T, 3), 
                'sbj_fullpose': net_output[..., 3:327].reshape(BS*T, 54, 2, 3),
                'sbj_global_orient':net_output[...,327:333].reshape(BS*T, 2, 3),
                'sbj_lr2omass': net_output[...,333:339].reshape(BS*T, 2, 3),
            }
            
            before_verts = self.prepare_recon(batch, net_outdict)['pred_verts']
            smplx_face = self.female_model.faces
            
            ### lr wrist joints stable refinement stage
            lr_joints_consis = 1e-4
            lr_lr2omass_refine = 1e-4

            upper_body_rotmat = nn.Parameter(net_outdict['sbj_fullpose'][:,12:21,...])
            
            opt_params_joints = {'sbj_smplx_transl': nn.Parameter(net_outdict['sbj_smplx_transl'].reshape(BS*T,3)), 
                               'sbj_global_orient': net_outdict['sbj_global_orient'].reshape(BS*T,2,3), 
                               'sbj_fullpose': nn.Parameter(net_outdict['sbj_fullpose']),
                               'sbj_lr2omass': nn.Parameter(net_outdict['sbj_lr2omass'].reshape(BS*T,2,3))}
            
            opt_joints_consis = optim.Adam([upper_body_rotmat], lr=lr_joints_consis)
            opt_lr2omass_refine = optim.Adam([opt_params_joints['sbj_lr2omass']], lr=lr_lr2omass_refine)
            
            
            num_iters_joints = 100 #if t!=0 else 600
            for itr in range(num_iters_joints):
                opt_joints_consis.zero_grad()
                opt_lr2omass_refine.zero_grad()
                
                recon_results = self.prepare_recon(batch, opt_params_joints)
                loss_total, losses, contact_label = self.calc_joints_consis_loss(batch, recon_results, opt_params_joints['sbj_lr2omass'])

                loss_total.backward()

                left_arms = [12,15,17,19]
                right_arms = [13,16,18,20]

                opt_grad = torch.zeros_like(upper_body_rotmat).to(self.device).to(torch.float32)
                opt_grad[:,[i-12 for i in left_arms],...] = contact_label[:,0].reshape(BS*T,1,1,1) * opt_params_joints['sbj_fullpose'].grad[:,left_arms,...]
                opt_grad[:,[i-12 for i in right_arms],...] = contact_label[:,1].reshape(BS*T,1,1,1) * opt_params_joints['sbj_fullpose'].grad[:,right_arms,...]
                
                upper_body_rotmat.grad = opt_grad

                opt_joints_consis.step()
                opt_lr2omass_refine.step()
                
                if itr % 10 == 0:
                    print(self.create_loss_message(losses, itr))
                    
            net_outdict['sbj_fullpose'][:,12:21,...] = upper_body_rotmat.detach().clone()
            
            net_outdict['sbj_smplx_transl'] = opt_params_joints['sbj_smplx_transl'].detach().clone()
            net_outdict['sbj_lr2omass'] = opt_params_joints['sbj_lr2omass'].detach().clone()
            
            if t==50:
                ### ground contact refinement stage
                lr_grd_contact = 1e-4
                opt_params_gnd = {'sbj_smplx_transl': nn.Parameter(net_outdict['sbj_smplx_transl'].reshape(BS*T,3)), 
                                    'sbj_global_orient': nn.Parameter(net_outdict['sbj_global_orient'].reshape(BS*T,2,3)), 
                                    'sbj_fullpose': nn.Parameter(net_outdict['sbj_fullpose'].reshape(BS*T,54,2,3))}
                
                opt_grd_contact = optim.Adam([opt_params_gnd[k] for k in ['sbj_smplx_transl']], lr=lr_grd_contact)
                num_iters_grn = 50
                for itr in range(num_iters_grn):
                    opt_grd_contact.zero_grad()
                    recon_results = self.prepare_recon(batch, opt_params_gnd)
                    loss_total, losses = self.calc_grd_contact_loss(batch, recon_results)
                    loss_total.backward()
                    opt_grd_contact.step()
                    if itr % 10 == 0:
                        print(self.create_loss_message(losses, itr))
                        
                for k, v in opt_params_gnd.items():
                    if k in ['sbj_smplx_transl']:
                        net_outdict[k] = v.detach().clone()
            
            
                ### lr contact and peneration refinement stage ###
                lr_hand_contact = 1e-4
                num_iters_contact = 100
                #### left hand
                left_hand_idx = [i+24 for i in range(15)]
                left_upper_body_idx = [15,17,19]
                left_upper_body_idx.extend(left_hand_idx) 
                # [15, 17, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
                l_hands_rotmat = [nn.Parameter(net_outdict['sbj_fullpose'][:,15:16,...]),
                                    nn.Parameter(net_outdict['sbj_fullpose'][:,17:18,...]),
                                    nn.Parameter(net_outdict['sbj_fullpose'][:,19:20,...]),
                                    nn.Parameter(net_outdict['sbj_fullpose'][:,24:39,...])]
                
                opt_params_contact = {'sbj_smplx_transl': net_outdict['sbj_smplx_transl'].reshape(BS*T,3), 
                                'sbj_global_orient': net_outdict['sbj_global_orient'].reshape(BS*T,2,3), 
                                'sbj_fullpose': nn.Parameter(net_outdict['sbj_fullpose'])}
                
                opt_lhand_obj_contact = optim.Adam(l_hands_rotmat, lr=lr_hand_contact)
                
                for itr in range(num_iters_contact):
                    opt_lhand_obj_contact.zero_grad()
                    recon_results = self.prepare_recon(batch, opt_params_contact)
                    loss_total, losses = self.calc_hand_obj_contact_loss(batch, recon_results, is_left = True)
                    if loss_total == 0:
                        print('[no contact]' + self.create_loss_message(losses, itr))
                        break
                    loss_total.backward()
                    l_hands_rotmat[2].grad = opt_params_contact['sbj_fullpose'].grad[:,19:20,...]
                    l_hands_rotmat[3].grad = opt_params_contact['sbj_fullpose'].grad[:,24:39,...]
                    opt_lhand_obj_contact.step()
                    if itr % 1 == 0:
                        print(self.create_loss_message(losses, itr))
                        
                net_outdict['sbj_fullpose'][:,19:20,...] = l_hands_rotmat[2].detach().clone()
                net_outdict['sbj_fullpose'][:,24:39,...] = l_hands_rotmat[3].detach().clone()
                    
                #### right hand
                right_hand_idx = [i+39 for i in range(15)]
                right_upper_body_idx = [16,18,20]
                right_upper_body_idx.extend(right_hand_idx)
                # [16, 18, 20, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
                r_hands_rotmat = [nn.Parameter(net_outdict['sbj_fullpose'][:,16:17,...]),
                                    nn.Parameter(net_outdict['sbj_fullpose'][:,18:19,...]),
                                    nn.Parameter(net_outdict['sbj_fullpose'][:,20:21,...]),
                                    nn.Parameter(net_outdict['sbj_fullpose'][:,39:54,...])]
                opt_params_contact = {'sbj_smplx_transl': net_outdict['sbj_smplx_transl'].reshape(BS*T,3), 
                                'sbj_global_orient': net_outdict['sbj_global_orient'].reshape(BS*T,2,3), 
                                'sbj_fullpose': nn.Parameter(net_outdict['sbj_fullpose'])}
                opt_rhand_obj_contact = optim.Adam(r_hands_rotmat, lr=lr_hand_contact)
                
                for itr in range(num_iters_contact):
                    opt_rhand_obj_contact.zero_grad()
                    recon_results = self.prepare_recon(batch, opt_params_contact)
                    loss_total, losses = self.calc_hand_obj_contact_loss(batch, recon_results,is_left = False)
                    if loss_total == 0:
                        print('[no contact]' + self.create_loss_message(losses, itr))
                        break
                    loss_total.backward()
                    r_hands_rotmat[2].grad = opt_params_contact['sbj_fullpose'].grad[:,20:21,...]
                    r_hands_rotmat[3].grad = opt_params_contact['sbj_fullpose'].grad[:,39:54,...]
                    opt_rhand_obj_contact.step()
                    if itr % 1 == 0:
                        print(self.create_loss_message(losses, itr))
                
                net_outdict['sbj_fullpose'][:,20:21,...] = r_hands_rotmat[2].detach().clone()
                net_outdict['sbj_fullpose'][:,39:54,...] = r_hands_rotmat[3].detach().clone()
                
            
            opt_results = {}
            opt_results['img'] = torch.cat([v.detach().clone().reshape(self.BS, self.T,-1) for v in net_outdict.values()], dim=2).permute(0,2,1).unsqueeze(1)
            self.latest_opt_results = opt_results
            return opt_results
    
    
    def calc_joints_consis_loss(self, batch, recon_results, lr2omass):
        
        new_joints_lr_position = self.recalcu_wrist_joints(batch, lr2omass.detach().clone()).detach().clone()

        losses = {}
        contact_label = batch['contact_label'].reshape(-1,2).to(self.device)
        
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
        
        return loss_total, losses, contact_label
    
    def calc_hand_obj_contact_loss(self, batch, recon_results, is_left):
        
        losses = {}
        contact_label = batch['contact_label'].reshape(-1,2).to(self.device)
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

    
    def calc_grd_contact_loss(self, batch, recon_results):
        
        losses = {}
        grnd_loss_weight = 1
        first_frame_xyz = recon_results['pred_joints'][0:1,60:66,:].detach().clone().repeat(30,1,1)
        frame_z = recon_results['pred_joints'][:,60:66,2]
        
        losses['grnd_loss'] = 0
        losses['grnd_loss'] += self.LossL2(first_frame_xyz, recon_results['pred_joints'][:,60:66,:])
        losses['grnd_loss'] += torch.mean(torch.abs(frame_z))
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

        # first frame contact detection
        first_contact_frames[0] = contact_label[0]
                
        for i in range(29):
            i = i + 1
            if contact_label_l[i-1] == 0 and contact_label_l[i] == 1:
                first_contact_frames[i][0] = 1
                
            if contact_label_r[i-1] == 0 and contact_label_r[i] == 1:
                first_contact_frames[i][1] = 1
        # sub_contact frame detection
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
        
    
    @staticmethod
    def create_loss_message(loss_dict, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'
    