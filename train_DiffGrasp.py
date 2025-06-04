import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch

from smplx import SMPLXLayer 
from smplx.lbs import batch_rodrigues

from datetime import datetime

from torch import nn, optim

import glob, time

import trimesh

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor, create_video
from loguru import logger

from bps_torch.bps import bps_torch


from omegaconf import OmegaConf

from losses import build_loss
from data.GRAB_dataloader import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from tqdm import tqdm

from tools.diff_optim_body import DiffOptim

from bmdm_models.diffusion_smpl import create_model_and_diffusion
from diffusion.resample import create_named_schedule_sampler

import functools

os.environ["WANDB_API_KEY"] = '168e9207044ee465f99a09cdaa69d6d46982cfb7'

cdir = os.path.dirname(sys.argv[0])

import wandb

class ModelRunner:

    def __init__(self,cfg, inference=False):
        
        self.cfg = cfg
        self.is_inference = inference
        
        wandb.init(
            mode='disabled',
            # resume=True,
            name=self.cfg.expr_ID,
            project="DiffGrasp",
            config={
            "expriment_id": self.cfg.expr_ID,
            "epochs": 1000,
            }
        )
        
        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)

        self.logger = logger.info
        self.logger('[%s] - Started training XXX, experiment code %s' % (cfg.expr_ID, starttime))
        self.logger('Torch Version: %s\n' % torch.__version__)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_brand = torch.cuda.get_device_name() if use_cuda else None
        gpu_count = cfg.num_gpus
        if use_cuda:
            self.logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

        self.load_data(cfg, inference)
        wandb.config.dataset = self.cfg.datasets.dataset_dir


        model_path = 'para_models/smplx'
        self.body_model = SMPLXLayer(
            model_path=model_path,
            gender='neutral',
            num_pca_comps=24,
            flat_hand_mean=True,
        ).to(self.device)

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

        # Create the network
        self.ddp_model, self.diffusion = create_model_and_diffusion(cfg.mdm)
        self.ddp_model = self.ddp_model.to(self.device)
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
        
        # Setup the training losses
        self.loss_setup()

        if cfg.num_gpus > 1:
            self.ddp_model = nn.DataParallel(self.ddp_model)
            self.logger("Training on Multiple GPU's")
        
        vars_network = [var[1] for var in self.ddp_model.named_parameters()]
        n_params = sum(p.numel() for p in vars_network if p.requires_grad)
        self.logger('Total Trainable Parameters for network is %2.2f M.' % ((n_params) * 1e-6))
        
        self.configure_optimizers()

        self.best_train_loss = np.inf
        self.best_val_loss = np.inf
        
        self.epochs_completed = 0
        self.cfg = cfg

        if inference and cfg.best_model is None:
            cfg.best_model = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
        if cfg.best_model is not None:
            self._get_network().load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
            self.logger('Restored trained model from %s' % cfg.best_model)

        self.bps_torch = bps_torch()

        # upload code 
        arti_code = wandb.Artifact('python', type='code')
        arti_code.add_file('train_DiffGrasp.py')
        arti_code.add_file('bmdm_models/diffusion_smpl.py')
        wandb.log_artifact(arti_code)

    def diff_optim_setup(self, visual_seqs, T):
            
        diff_optim = DiffOptim(BS=visual_seqs, T=T, obj_sdfs=None, cfg = self.cfg)
        return diff_optim
    

    def loss_setup(self):

        self.logger('Configuring the losses!')
        self.LossL2 = nn.MSELoss(reduction='mean')
        
        self.vertex_loss_weight = 200
        wandb.config.vertex_loss_weight = self.vertex_loss_weight       
        self.diff_loss_weight = 100
        wandb.config.diff_loss_weight = self.diff_loss_weight
        self.interaction_loss_weight = 100
        wandb.config.interaction_loss_weight = self.interaction_loss_weight
        self.lrj2omass_loss_weight = 100
        wandb.config.lrj2omass_loss_weight = self.lrj2omass_loss_weight
        
    def load_data(self,cfg, inference):

        self.logger('Base dataset_dir is %s' % self.cfg.datasets.dataset_dir)

        ds_dir = self.cfg.datasets.dataset_dir
        idx2obj_path = os.path.join(ds_dir, 'idx2obj.pickle')
        import pickle 
        self.idx2obj = pickle.load(open(idx2obj_path, 'rb'))
        self.obj_mesh = {}
        for k,v in self.idx2obj.items():
            self.obj_mesh[k] = trimesh.load(os.path.join(ds_dir,'obj_meshes','{}.obj'.format(v)), process=False)
            
        ds_name = 'test'
        ds_test = LoadData(self.cfg.datasets, split_name=ds_name)
        self.ds_test = build_dataloader(ds_test, split=ds_name, cfg=self.cfg.datasets)

        if not inference:

            ds_name = 'train'
            ds_train = LoadData(self.cfg.datasets, split_name=ds_name)
            self.ds_train = build_dataloader(ds_train, split=ds_name, cfg=self.cfg.datasets)

            ds_name = 'val'
            ds_val = LoadData(self.cfg.datasets, split_name=ds_name)
            self.ds_val = build_dataloader(ds_val, split=ds_name, cfg=self.cfg.datasets)

        if not inference:
            self.logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
                        (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

    def _get_network(self):
        return self.ddp_model.module if isinstance(self.ddp_model, torch.nn.DataParallel) else self.ddp_model

    def save_network(self):
        torch.save(self.ddp_model.module.state_dict()
                   if isinstance(self.ddp_model, torch.nn.DataParallel)
                   else self.ddp_model.state_dict(), self.cfg.best_model)

    def forward_loss(self, batch, batch_idx, type, cfg):

        BS, T, _, _, _ = batch['sbj_fullpose'].shape

        x_0 = {}
        ''''
        sbj_smplx_transl   torch.Size([16, 30, 3])
        sbj_fullpose       torch.Size([16, 30, 54, 3, 3])
        sbj_global_orient  torch.Size([16, 30, 3, 3])
        sbj_lr2omas        torch.Size([16, 30, 2, 3])
        '''
        x_0['sbj_smplx_transl'] = batch['sbj_smplx_transl']
        x_0['sbj_fullpose'] = batch['sbj_fullpose'][:, :, :, :2, :]
        x_0['sbj_global_orient'] = batch['sbj_global_orient'][:, :, :2, :]
        x_0['sbj_lr2omas'] = batch['sbj_lr2omas'] 
        
        x_0 = torch.cat([v.reshape(BS, T,-1).to(self.device) for v in x_0.values()], dim=2)
        x_0 = x_0.permute(0,2,1).unsqueeze(1).contiguous() # torch.Size([BS, 1, 317, 60])
        
        ### CONDITION
        
        embed_condition = self.ddp_model._get_cond_embeddings_align(batch, self.device)
        # exit()
        model_kwargs = {'y': {'cond': embed_condition}}
        # sample random t and weights(uniforms)
        t, weights = self.schedule_sampler.sample(BS, self.device)

        # forward
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            x_0,  # [BS, ch, image_size, image_size]
            t,  # [BS](int) sampled timesteps
            model_kwargs=model_kwargs,
        )
        pred, gt = compute_losses()

        pred = pred.squeeze(1).permute(0,2,1).contiguous()
        gt = gt.squeeze(1).permute(0,2,1).contiguous()
        diff_gened = {'pred': pred, 'gt': gt}
        smplx_pred = {
            'sbj_smplx_transl': pred[..., 0:3].reshape(BS, T, 3),
            'sbj_fullpose': pred[..., 3:327].reshape(BS, T, 54, 2, 3),
            'sbj_global_orient': pred[...,327:333].reshape(BS, T, 2, 3),
            'sbj_lr2omass': pred[...,333:339].reshape(BS, T, 2, 3)
        }
        recon_results = self.recon_smplx(batch, smplx_pred)
        loss_total, losses = self.get_loss(batch, diff_gened, recon_results)

        # loss_total, losses = self.get_loss_siggraphasia(batch = batch, diff_gened = diff_gened, recon_out = recon_out)
        
        return loss_total, losses

    
    
    def sample(self, batch, is_vis, it, ds_name):
        
        BS, T, _, _, _ = batch['sbj_fullpose'].shape

        x_0 = {}
        ''''
        sbj_smplx_transl   torch.Size([16, 30, 3])
        sbj_fullpose       torch.Size([16, 30, 54, 3, 3])
        sbj_global_orient  torch.Size([16, 30, 3, 3])
        sbj_lr2omas        torch.Size([16, 30, 2, 3])
        '''
        x_0['sbj_smplx_transl'] = batch['sbj_smplx_transl']
        x_0['sbj_fullpose'] = batch['sbj_fullpose'][:, :, :, :2, :]
        x_0['sbj_global_orient'] = batch['sbj_global_orient'][:, :, :2, :]
        x_0['sbj_lr2omas'] = batch['sbj_lr2omas'] 
        
        x_0 = torch.cat([v.reshape(BS, T,-1).to(self.device) for v in x_0.values()], dim=2)
        x_0 = x_0.permute(0,2,1).unsqueeze(1).contiguous() # torch.Size([BS, 1, 317, 60])

        # transformer encoder (encoding condition)
        embed_condition = self.ddp_model._get_cond_embeddings_align(batch, self.device)
        
        model_kwargs = {'y': {'cond': embed_condition}}
        model_kwargs['y']['inpainted_motion'] = x_0
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(x_0, dtype=torch.bool,
                                                                    device=self.device)  # True means use gt motion
        model_kwargs['y']['inpainting_mask'][:, :, :, :] = False
        

        # forward
        diff_optim = self.diff_optim_setup(visual_seqs=1, T=30)
        diff_optim.recon_obj_meshes(batch = batch, idx2obj = self.idx2obj, it = it)
        
        sample_fn = self.diffusion.p_sample_loop
        dump_steps = [1000-i for i in dump_steps]

        pred_1000, direct_denoise = sample_fn(batch, self.ddp_model, diff_optim, x_0.shape, clip_denoised=False, progress=True, model_kwargs=model_kwargs,
                         dump_steps=dump_steps) 
        
        diff_loss = {'pred': pred_1000, 'gt': x_0}
        
        pred_1000 = pred_1000.squeeze(1).permute(0,2,1).contiguous()
        
        results_1000 = {
            'sbj_smplx_transl': pred_1000[..., 0:3].reshape(BS, T, 3),
            'sbj_fullpose': pred_1000[..., 3:327].reshape(BS, T, 54, 2, 3),
            'sbj_global_orient': pred_1000[...,327:333].reshape(BS, T, 2, 3),
            'sbj_lr2omass': pred_1000[...,333:339].reshape(BS, T, 2, 3)
        }        

        results_1000 = self.recon_smplx(batch, results_1000, is_recon_gt=True)
        self.visualize_seqs(1, results_1000, batch, it, ds_name,additional_name='all')

        return diff_loss, results_1000
    
    def recon_smplx(self, batch, sampled_results, is_recon_gt = False):
        
        recon_output = {}
        pred_poses = torch.cat((sampled_results['sbj_global_orient'].unsqueeze(2), sampled_results['sbj_fullpose']), dim=2)
        pred_results = self.smplxLayer(batch, poses=pred_poses, trans=sampled_results['sbj_smplx_transl'], is_6D=True)
        recon_output['pred_verts'] = pred_results['verts']
        recon_output['pred_joints'] = pred_results['joints']
        
        if is_recon_gt:
            gt_results = self.smplxLayer(batch, poses=torch.cat((batch['sbj_global_orient'].unsqueeze(2), batch['sbj_fullpose']),dim=2),
                                        trans=batch['sbj_smplx_transl'], is_6D=False)
            recon_output['gt_verts'] = gt_results['verts']
            recon_output['gt_joints'] = gt_results['joints']
            
        recon_output['sampled_results'] = sampled_results
        
        return recon_output
        
    def train(self):
        
        self.ddp_model.train()
        is_vis = True 
        train_loss_dict = {}
        lr_tmp = self.optimizer.param_groups[0]['lr']
        for it, batch in tqdm(enumerate(self.ds_train), total=len(self.ds_train)):
            
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            loss_total, losses_dict = self.forward_loss(batch, it, type=None, cfg=None)

            loss_total.backward()
            nn.utils.clip_grad_norm_(self.ddp_model.parameters(), 10)
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}
            
            
            cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
            train_msg = self.create_loss_message(cur_train_loss_dict,
                                                expr_ID=self.cfg.expr_ID,
                                                epoch_num=self.epochs_completed,
                                                model_name='diffu_grasp',
                                                it=it,
                                                try_num=0,
                                                mode='train',
                                                lr = lr_tmp)

            self.logger(train_msg)
            
            wandb.log({"train_diffusion_loss": cur_train_loss_dict['loss_diffu'], 
                        "train_loss_pose_diff": cur_train_loss_dict['loss_pose_diff'],
                        "train_lrj2omass_loss": cur_train_loss_dict['lrj2omass_loss'],
                        "train_recon_joints": cur_train_loss_dict['recon_pose_joints'],
                        "train_interaction_loss": cur_train_loss_dict['interaction'],
                        "train_total_loss": cur_train_loss_dict['loss_total'],
                        })
            
        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        
        return train_loss_dict
            

    def evaluate(self, ds_name='val'):
        self.ddp_model.eval()

        eval_loss_dict = {}

        data = self.ds_val

        with torch.no_grad():
            for it, batch in tqdm(enumerate(data), total=len(data)):
               
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()
                loss_total, losses_dict = self.forward_loss(batch, it, type=None, cfg=None)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}
                
                cur_val_loss_dict = {k: v / (it + 1) for k, v in eval_loss_dict.items()}

                wandb.log({"val_diffusion_loss": cur_val_loss_dict['loss_diffu'], 
                        "val_loss_pose_diff": cur_val_loss_dict['loss_pose_diff'],
                        "val_lrj2omass_loss": cur_val_loss_dict['lrj2omass_loss'],
                        "val_recon_joints": cur_val_loss_dict['recon_pose_joints'],
                        "val_interaction_loss": cur_val_loss_dict['interaction'],
                        "val_total_loss": cur_val_loss_dict['loss_total'],
                        })

            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}

        return eval_loss_dict
    
    def infer(self, ds_name='test'):
        self.ddp_model.eval()
        data = self.ds_test

        with torch.no_grad():
            for it, batch in tqdm(enumerate(data), total=len(data)):
                is_vis = True 
                diff_loss, results_1000 = self.sample(batch, is_vis, it, ds_name)


    def get_loss(self, batch, diff_gened, recon_out):   

        sampled_results = recon_out['sampled_results']
        
        losses = {}
        batch_joints = batch['sbj_smplx_joints'].reshape(-1,127,3).to(self.device)

        # masked hand joints loss
        losses['recon_pose_joints'] = 0
        contact_label = batch['contact_label'].reshape(-1,2)
        contact_label = torch.cat((contact_label[:,0].reshape(-1,1).repeat(1,21),contact_label[:,1].reshape(-1,1).repeat(1,21)), dim=1) # BS*T, 42
        lr_joints_pred = torch.cat((recon_out['pred_joints'][:,20:21,:],recon_out['pred_joints'][:,25:40,:], recon_out['pred_joints'][:,66:71,:],
                                  recon_out['pred_joints'][:,21:22,:],recon_out['pred_joints'][:,40:55,:], recon_out['pred_joints'][:,71:76,:]),dim=1)
        lr_joints_gt = torch.cat((batch_joints[:,20:21], batch_joints[:,25:40,:], batch_joints[:,66:71,:],
                                  batch_joints[:,21:22,:], batch_joints[:,40:55,:], batch_joints[:,71:76,:]),dim=1) # l 0:20 r:20:
        joints_dist = torch.norm(lr_joints_pred - lr_joints_gt, dim=2)
        recon_masked_joints_loss =  torch.mean(contact_label.to(self.device) * joints_dist)
        losses['recon_pose_joints'] += recon_masked_joints_loss
        # other body joints
        losses['recon_pose_joints'] += self.LossL2(recon_out['pred_joints'][:,0:20,:], batch_joints[:,0:20,:])
        losses['recon_pose_joints'] *= self.vertex_loss_weight
        
        # diffusion loss
        if diff_gened is not None:
            losses['loss_diffu'] = 0
            losses['loss_diffu'] += self.LossL2(diff_gened['pred'], diff_gened['gt'])
            losses['loss_diffu'] *= self.diff_loss_weight
        
        losses['loss_pose_diff'] = 0
        losses['loss_pose_diff'] += self.LossL2(diff_gened['pred'][...,3:327], diff_gened['gt'][...,3:327])
        losses['loss_pose_diff'] *= self.diff_loss_weight

        # rel lrj loss
        losses['lrj2omass_loss'] = 0
        contact_label = batch['contact_label'].reshape(-1,2) # BS*T, 2
        lrj2omass_dis = torch.norm(sampled_results['sbj_lr2omass'].reshape(-1,2,3)- batch['sbj_lr2omas'].reshape(-1,2,3).to(self.device),dim=2)
        losses['lrj2omass_loss'] += torch.mean(contact_label.to(self.device) * lrj2omass_dis) *2
        losses['lrj2omass_loss'] *= self.vertex_loss_weight
        
        ### interaction_loss
        losses['interaction'] = 0
        # term 1 for body joints
        obj_mass_term1 = batch['obj_mass'].reshape(-1,1,3).repeat(1,22,1).to(self.device)
        dist1 = torch.norm(obj_mass_term1-recon_out['pred_joints'][:,:22,:],dim=2)
        dist2 = torch.norm(obj_mass_term1-batch_joints[:,:22,:],dim=2)
        losses['interaction'] += self.LossL2(dist1, dist2)
        # term 2 for hand joints
        contact_label = batch['contact_label'].reshape(-1,2)
        contact_label = torch.cat((contact_label[:,0].reshape(-1,1).repeat(1,20),contact_label[:,1].reshape(-1,1).repeat(1,20)), dim=1) # BS*T, 40
        
        obj_mass_term2 = batch['obj_mass'].reshape(-1,1,3).repeat(1,40,1).to(self.device)
        lr_joints_gt = torch.cat((batch_joints[:,25:40,:], batch_joints[:,66:71,:],
                                  batch_joints[:,40:55,:], batch_joints[:,71:76,:]),dim=1) # l 0:20 r:20:
        dist_lambda = 2
        dist_weight = torch.exp(-(dist_lambda * torch.norm(obj_mass_term2 - lr_joints_gt, dim=2))) # BS*T, 40
        masked_dist_weight = contact_label.to(self.device) * dist_weight #BS*T,40
        
        lr_joints_pred = torch.cat((recon_out['pred_joints'][:,25:40,:], recon_out['pred_joints'][:,66:71,:],
                                  recon_out['pred_joints'][:,40:55,:], recon_out['pred_joints'][:,71:76,:]),dim=1) # l 0:20 r:20:
        pred_dist = torch.norm(obj_mass_term2 - lr_joints_pred, dim=2) # BS*T, 40

        losses['interaction'] += torch.mean(masked_dist_weight * pred_dist)
        losses['interaction'] *= self.interaction_loss_weight   

        loss_total = torch.stack(list(losses.values())).sum()
        
        losses['loss_total'] = loss_total
        
        return loss_total, losses
    
    def smplxLayer(self, batch, poses, trans, is_6D):

        BS = poses.shape[0]
        T = poses.shape[1]
        poses = poses.reshape(BS*T,55,-1,3).to(self.device)
        trans = trans.reshape(BS*T,3).to(self.device)
        bparams = parms_6D2full(poses, trans, d62rot=is_6D)
        
        genders = batch['gender'].reshape(-1)
        males = genders == 1
        females = ~males
        
        v_template = batch['smplx_v_temp'].reshape(-1,10475,3).to(self.device)
        

        FN = sum(females)
        MN = sum(males)
        
        recon_output = {}
        recon_output['verts'] = torch.zeros(BS*T,10475,3).to(self.device)
        recon_output['joints'] = torch.zeros(BS*T,127,3).to(self.device)
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
    
    def visualize_seqs(self, visual_seqs,sampled_results, batch, it, ds_name, additional_name=''):
        with torch.no_grad():
            
            BS = batch['sbj_smplx_transl'].shape[0]
            T = batch['sbj_smplx_transl'].shape[1]   
            
            pred_verts = sampled_results['pred_verts'].reshape(BS, T, 10475, 3)[:visual_seqs,...]
            pred_joints = sampled_results['pred_joints'].reshape(BS, T, 127, 3)[:visual_seqs,...]
            gt_verts = sampled_results['gt_verts'].reshape(BS, T, 10475, 3)[:visual_seqs,...]
            gt_joints = sampled_results['gt_joints'].reshape(BS, T, 127, 3)[:visual_seqs,...]
            

            for seq_idx in range(visual_seqs):
                obj_name = self.idx2obj[batch['obj_idx'][seq_idx,0].item()]
                obj_path = os.path.join(self.cfg.datasets.dataset_dir, 'obj_meshes', obj_name+'.obj')
                obj_temp = trimesh.load(obj_path, process=False)

                obj_verts = torch.from_numpy(obj_temp.vertices).to(torch.float32)
                obj_verts = obj_verts.repeat(30,1,1)
                obj_verts_rot = torch.matmul(obj_verts, batch['obj_global_orient'][seq_idx,...]) + batch['obj_transl'][seq_idx,...].unsqueeze(1)

                gender = 1 if batch['gender'][seq_idx][0] == 1 else 0
                if gender == 1:
                    human_face = self.male_model.faces
                else:
                    human_face = self.female_model.faces
                
                save_dict = {'smplx':{
                                    'beta': to_cpu(batch['beta'][0,0]), # 10
                                    'global_orient':to_cpu(d62rotmat(sampled_results['sampled_results']['sbj_global_orient'].reshape(BS,T,2,3)[seq_idx]).reshape(30,-1)), # 30,3,3
                                    'transl': to_cpu(sampled_results['sampled_results']['sbj_smplx_transl'].reshape(BS,T,3)[seq_idx].reshape(30,3)),# 30,3
                                    'body_pose': to_cpu(d62rotmat(sampled_results['sampled_results']['sbj_fullpose'].reshape(BS,T,54,2,3)[seq_idx,:,0:21,...]).reshape(30,-1)), # 30,-1
                                    'left_hand_pose': to_cpu(d62rotmat(sampled_results['sampled_results']['sbj_fullpose'].reshape(BS,T,54,2,3)[seq_idx,:,24:39,...]).reshape(30,-1)), # 30,-1
                                    'right_hand_pose': to_cpu(d62rotmat(sampled_results['sampled_results']['sbj_fullpose'].reshape(BS,T,54,2,3)[seq_idx,:,39:,...]).reshape(30,-1)), # 30,-1
                                    'gender':gender, # 1
                                    },
                            'obj':{
                                    'vertice': to_cpu(obj_verts_rot), # 30,-1,3
                                    'faces': obj_temp.faces # -1,3
                                    },
                            'human':{
                                    'vertice': to_cpu(pred_verts[seq_idx]), # 30,10475,3
                                    'joints': to_cpu(pred_joints[seq_idx]), # 30,127,3
                                    'faces':human_face # -1,3
                                    }
                            }
                save_dir_path = os.path.join('cal_metrics/',self.cfg.expr_ID, str(it*BS + seq_idx)+'-0', 'pred')
                os.makedirs(save_dir_path, exist_ok=True)
                save_path = os.path.join(save_dir_path, 'result.npy')
                with open(save_path,'wb') as f:
                    np.save(f, save_dict)


                #### save gt
                save_dict_gt = {'smplx':{
                                    'beta': to_cpu(batch['beta'][0,0]), # 10
                                    'global_orient':to_cpu(batch['sbj_global_orient'].reshape(BS, T, 3, 3)[seq_idx].reshape(30,3,3)), # 30,3,3
                                    'transl': to_cpu(batch['sbj_smplx_transl'].reshape(BS, T, 3)[seq_idx].reshape(30,3)),# 30,3
                                    'body_pose': to_cpu(batch['sbj_fullpose'].reshape(BS, T, 54 ,3 ,3)[seq_idx,:,0:21,...].reshape(30,-1)), # 30,-1
                                    'left_hand_pose': to_cpu(batch['sbj_fullpose'].reshape(BS, T, 54 ,3 ,3)[seq_idx,:,24:39,...].reshape(30,-1)), # 30,-1
                                    'right_hand_pose': to_cpu(batch['sbj_fullpose'].reshape(BS, T, 54 ,3 ,3)[seq_idx,:,39:,...].reshape(30,-1)), # 30,-1
                                    'gender':gender, # 1
                                    },
                            'obj':{
                                    'vertice': to_cpu(obj_verts_rot), # 30,-1,3
                                    'faces': obj_temp.faces # -1,3
                                    },
                            'human':{
                                    'vertice': to_cpu(gt_verts[seq_idx]), # 30,10475,3
                                    'joints': to_cpu(gt_joints[seq_idx]), # 30,127,3
                                    'faces':human_face # -1,3
                                    }
                            }
                
                save_dir_path = os.path.join('cal_metrics/',self.cfg.expr_ID , str(it*BS + seq_idx)+'-0', 'gt')
                os.makedirs(save_dir_path, exist_ok=True)
                save_path = os.path.join(save_dir_path, 'result.npy')
                with open(save_path,'wb') as f:
                    np.save(f, save_dict_gt)
                if it % 20==0:
                    mesh_obj = {'verts': to_cpu(obj_verts_rot), 'faces': obj_temp.faces}
                    mesh_pred = {'verts': to_cpu(pred_verts[seq_idx]), 'faces': human_face}
                    mesh_gt = {'verts': to_cpu(gt_verts[seq_idx]), 'faces': human_face}
                    animation = {'obj':mesh_obj, 'pred':mesh_pred, 'gt':mesh_gt}

                    animation_path = os.path.join(self.cfg.work_dir, 'visualization/', 'epoch_'+str(self.epochs_completed),ds_name)
                    if not os.path.exists(animation_path): os.makedirs(animation_path)
                    np.save(os.path.join(animation_path, additional_name+str(it)+'_'+str(seq_idx)+'.npy'), animation)

                    self.logger('visualization done; npy files had saved in dir %s ' % animation_path)
            

    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr = np.inf

        for epoch_num in range(1, n_epochs + 1):
            if self.epochs_completed > n_epochs:
                break
            
            if self.cfg.is_continue == True:
                
                best_model = sorted(glob.glob(os.path.join(self.cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
                self._get_network().load_state_dict(torch.load(best_model, map_location=self.device), strict=False)
                import re
                pattern = r"(\d{3})_model\.pt"
                self.epochs_completed = int(re.search(pattern, best_model).group(1))+1
                
                self.logger('--- load previous best model # ' + best_model)
                self.cfg.is_continue = False
                
            self.logger('--- starting Epoch # %03d' % self.epochs_completed)            
            
            train_loss_dict = self.train()  

            eval_loss_dict  = self.evaluate()

            self.lr_scheduler.step(train_loss_dict['loss_total'])
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            with torch.no_grad():
                
                wandb.log({"epoch": self.epochs_completed,
                            'lr':cur_lr})
                
                if train_loss_dict['loss_total'] < self.best_train_loss or eval_loss_dict['loss_total'] < self.best_val_loss:
                    self.cfg.best_model = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'E%03d_model.pt' % (self.epochs_completed)), isfile=True)
                    self.save_network()
                if train_loss_dict['loss_total'] < self.best_train_loss:
                    self.best_train_loss = train_loss_dict['loss_total']
                if eval_loss_dict['loss_total'] < self.best_val_loss:
                    self.best_val_loss = eval_loss_dict['loss_total']
                    
            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)
        wandb.finish()
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_train_loss))
        self.logger('Best model path: %s\n' % self.cfg.best_model)
        

    def inference(self):

        starttime = datetime.now().replace(microsecond=0)
            
        best_model = sorted(glob.glob(os.path.join(self.cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
        self._get_network().load_state_dict(torch.load(best_model, map_location=self.device), strict=False)
        import re
        pattern = r"(\d{3})_model\.pt"
        self.epochs_completed = int(re.search(pattern, best_model).group(1))+1
        
        self.logger('--- load previous best model # ' + best_model)
        self.cfg.is_continue = False
                
        self.logger('--- inference Epoch # %03d' % self.epochs_completed)                

        self.infer()

        endtime = datetime.now().replace(microsecond=0)
        wandb.finish()
        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_train_loss))
        self.logger('Best model path: %s\n' % self.cfg.best_model)


    def configure_optimizers(self):

        self.optimizer = optim.AdamW(self.ddp_model.parameters(),lr=1e-4,betas=(0.5, 0.999))

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=8)
        
    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='mlp', it=0, try_num=0, mode='evald', lr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s:  - lr [%s] - %s ' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, lr, ext_msg)


def run():

    import argparse
    from configs.DiffGrasp_config import conf as cfg

    parser = argparse.ArgumentParser(description='GNet-Training')

    parser.add_argument('--work-dir',
                        default='work_dir/',
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--expr-id', default='DiffGrasp', type=str,
                        help='Training ID')

    parser.add_argument('--is-continue', default=False,
                        type=int,
                        help='Training from the last breakpoint')
    
    parser.add_argument('--num-gpus', default=1,
                        type=int,
                        help='Number of multiple GPUs for training')
    
    parser.add_argument('--mode', default='training',
                    type=str,
                    help='Selecting model states(training/inference).')

    cmd_args = parser.parse_args()

    cfg.expr_ID = cfg.expr_ID if cmd_args.expr_id is None else cmd_args.expr_id    
    cfg.output_folder = cmd_args.work_dir
    cfg.num_gpus = cmd_args.num_gpus
    cfg.is_continue = cmd_args.is_continue
    cfg.mode = cmd_args.mode

    cfg.work_dir = os.path.join(cfg.output_folder, cfg.expr_ID)
    makepath(cfg.work_dir)
    
    ########################################
    
    run_runner_once(cfg)

def run_runner_once(cfg):
    
    runner = ModelRunner(cfg=cfg)
    OmegaConf.save(runner.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))
    if cfg.mode == 'training':
        runner.fit(n_epochs=500)
    if cfg.mode == 'inference':
        runner.inference()

    OmegaConf.save(runner.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))

if __name__ == '__main__':

    run()
