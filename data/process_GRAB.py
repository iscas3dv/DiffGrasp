import sys
sys.path.append('')
sys.path.append('..')
import numpy as np
import torch
import os, shutil, glob

from datetime import datetime
from tqdm import tqdm

from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config
# from tools.meshviewer import Mesh

from tools.utils import makepath, makelogger
from tools.utils import parse_npz
from tools.utils import to_cpu, to_np, to_tensor
from tools.utils import np2torch, torch2np
from tools.utils import aa2rotmat, rotmat2aa, rotate, rotmul, euler
from models.model_utils import full2bone, full2bone_aa, parms_6D2full

from bps_torch.bps import bps_torch

from psbody.mesh.colors import name_to_rgb
from psbody.mesh import Mesh, MeshViewers, MeshViewer

from tools.vis_tools import sp_animation

from smplx import SMPLXLayer 
from smplx.lbs import batch_rodrigues
import random
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArcticDataSet(object):
    def __init__(self, cfg = None):
        
        self.cwd = os.path.dirname(sys.argv[0])
        self.bps_torch = bps_torch()
        bps_orig_path = f'{self.cwd}/../configs/bps.pt'
        self.bps = torch.load(bps_orig_path)
        
        ### Change to your datapath
        self.root_path = '/data/heqiang/cvpr2025/dataset/GRAB'
        self.out_path = '/data/heqiang/GRAB_diffgrasp_test'
        
        self.seq_path = os.path.join(self.root_path, 'grab')
        self.tools_path = os.path.join(self.root_path, 'tools')

        
        self.total_lens = 0 
        self.all_seqs = []
        self.all_seqs = glob.glob(os.path.join(self.seq_path ,'*/*.npz'))
        
        import lmdb
        lmdb_path = self.out_path
        lmdb_env = lmdb.open(lmdb_path, map_size=987842478080)  
        
        self.all_objs = []
        for i in range(len(self.all_seqs)):
            obj_t = self.all_seqs[i].split('/')[-1].split('_')[0]
            self.all_objs.append(obj_t)

        ### all object for training
        ### without 5 object (apple, toothbrush, train, cubelarge, phone) for unknown object testing
        self.object_list = ['cylinderlarge', 'mug', 'elephant', 'stanfordbunny', 'airplane', 'alarmclock', 'banana', 'body', 'bowl', 'cubesmall', 'cup', 'cubemedium',
             'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer', 'headphones', 'knife', 'lightbulb', 'mouse', 'piggybank', 'pyramidlarge', 'pyramidsmall', 'pyramidmedium',
             'duck', 'scissors', 'spherelarge', 'spheresmall', 'stamp', 'stapler', 'teapot', 'toruslarge', 'torussmall', 'watch', 'cylindersmall', 'waterbottle', 'torusmedium',
             'cylindermedium', 'spheremedium', 'wristwatch', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste','doorknob', 'hand']
        self.object_list = ['apple']
        
        # Move object meshes to a new folder
        for obj in self.object_list:
            obj_mesh_path = makepath(os.path.join(self.out_path, 'obj_meshes', '%s.obj' % obj), isfile=True)
            obj_mesh = Mesh(filename=os.path.join(self.tools_path, 'object_meshes/processed_object_meshes', obj+'.ply'))
            obj_mesh.write_obj(obj_mesh_path)
        
        idx2obj = {i: self.object_list[i] for i in range(len(self.object_list))}
        obj2idx = {value: key for key, value in idx2obj.items()}
        
        # Move subject meshes to a new folder
        sbj_src_path = makepath(os.path.join(self.tools_path, 'subject_meshes'))
        sbj_out_path = makepath(os.path.join(self.out_path, 'subject_meshes'))
        for root, dirs, files in os.walk(sbj_src_path):
            rel_path = os.path.relpath(root, sbj_src_path)
            target_root = os.path.join(sbj_out_path, rel_path) if rel_path != '.' else sbj_out_path
            os.makedirs(target_root, exist_ok=True)
            
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_root, file)
                shutil.copy2(src_file, dst_file)
        ### split dataset by subjects
        self.train_sbj = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
        self.val_sbj =  ['s9']
        self.test_sbj =  ['s10']

        self.para_models_path = '/data/heqiang/diffgrasp/para_models'
        model_path = os.path.join(self.para_models_path, 'smplx')
        self.female_model = SMPLXLayer(
                            model_path=model_path,
                            gender='female',
                            num_pca_comps=24,
                            # flat_hand_mean=True,
                        )
        self.male_model = SMPLXLayer(
                            model_path=model_path,
                            gender='male',
                            num_pca_comps=24,
                            # flat_hand_mean=True,
                        )

        correspondence_path = f'{self.cwd}/../configs/MANO_SMPLX_vertex_ids.pkl'
        with open(correspondence_path, 'rb') as f:
            import pickle
            idxs_data = pickle.load(f)

        self.smplx_mano_left_verts_bool = torch.zeros(10475, dtype=torch.bool)
        self.smplx_mano_right_verts_bool = torch.zeros(10475, dtype=torch.bool)
        self.smplx_mano_left_verts_bool[idxs_data['left_hand']] = True
        self.smplx_mano_right_verts_bool[idxs_data['right_hand']] = True

  
        self.total_lens = 0
        
        ### welford algorithm: Calculate the mean and std iteratively
        ds_welford = {'pose': {'count': None, 'mean': None, 'm2':None},
                   'rot2world': {'count': None, 'mean': None, 'm2':None},
                   'transl2world': {'count': None, 'mean': None, 'm2':None},
                   'joints': {'count': None, 'mean': None, 'm2':None},
                   'lr_2omass': {'count': None, 'mean': None, 'm2':None},
                   }
        ds_id = {'train':0,'test':0,'val':0}
        
        
        ### process sequence by sequence
        for seq_i in tqdm(range(len(self.all_seqs))):  
            torch.cuda.empty_cache()     
            obj_name = self.all_seqs[seq_i].split('/')[-1].split('_')[0]
            print(self.all_seqs[seq_i])
            if obj_name not in self.object_list:
                continue
            seq_data = parse_npz(self.all_seqs[seq_i])
            
            smplx_seq_params = seq_data['body']['params']
            sub_gender = seq_data['gender']
            sid = seq_data['sbj_id']

            self.sbj_info = {}
            self.obj_info = {}
            
            v_templates = os.path.join(self.root_path, seq_data['body']['vtemp'])
            v_templates = Mesh(filename=v_templates)

            gender = 1 if sub_gender == 'male' else 0
            beta = os.path.join(self.tools_path, 'subject_meshes', sub_gender, sid+'_betas.npy')
            beta = np.load(beta)
            
            obj_seq_params = seq_data['object']['params']
            obj_mesh = Mesh(filename=os.path.join(self.root_path, seq_data['object']['object_mesh']))
            
            contact_body = seq_data['contact']['body']
            
            intervals, seqs_contact_label = self.sum_contact_frames(contact_body)
            
            ### sliding window 
            frame_stride = 10
            fix_gen_frame = 30
            window = np.array([i for i in range(0,fix_gen_frame*frame_stride,frame_stride)])
            
            
            inter_start_idx = intervals[0]
            inter_end_idx = intervals[1]

            windows = []
            for i in range(inter_start_idx, inter_end_idx - window[-1],17):
                sliding_window = window + i
                windows.append(sliding_window)


            ds_name = ''
            if sid in self.train_sbj:
                ds_name = 'train'    
            elif sid in self.val_sbj:
                ds_name = 'val'
            else:
                ds_name = 'test'
            
            self.total_lens += len(windows)
            
            with torch.no_grad():
                if len(windows) == 0:
                    continue
                windows_idxs = np.array(windows).reshape(-1)
                
                
                # SMPLX_params
                smplx_transl = torch.tensor(smplx_seq_params['transl'][windows_idxs]).reshape(-1,30,3)
                
                rel_transl = smplx_transl.reshape(-1,30,3)[:,0:1,:].clone()
                rel_transl[...,2:3] = torch.zeros_like(rel_transl[...,2:3])
                rel_transl = rel_transl.repeat(1,30,1)
                
                smplx_transl = (smplx_transl.reshape(-1,30,3) - rel_transl).reshape(-1,3)
                
                smplx_global_orient = torch.tensor(smplx_seq_params['global_orient'][windows_idxs])
                
                smplx_full_pose = torch.tensor(smplx_seq_params['fullpose'][windows_idxs,3:]).reshape(-1,54,3)

                if gender == 1:
                    self.male_model.v_template = torch.tensor(v_templates.v).to(torch.float32)
                    smplx_model = self.male_model.to(device)
                    
                else:
                    self.female_model.v_template = torch.tensor(v_templates.v).to(torch.float32)
                    smplx_model = self.female_model.to(device)
                    
                smplx_pose_aa = smplx_full_pose
                smplx_pose_rotmat = aa2rotmat(smplx_full_pose)
                
                smplx_global_aa = smplx_global_orient
                smplx_global_rotmat = aa2rotmat(smplx_global_orient).squeeze(1)
                
                smplx_pose_rotmat_forward = torch.cat((smplx_global_rotmat.unsqueeze(1),smplx_pose_rotmat), dim=1)
                
                smplx_pose_params = parms_6D2full(smplx_pose_rotmat_forward.reshape(-1, 55,3,3),
                                    smplx_transl.reshape(-1 ,3),
                                    d62rot=False)
                smplx_pose_params = {k: v.to(device) for k, v in smplx_pose_params.items()}
 
                
                output_gt = smplx_model(**smplx_pose_params)
                # smplx_vertices = output_gt.vertices 
                smplx_joints = output_gt.joints.cpu()

                ### obj processing
                obj_transl = torch.tensor(obj_seq_params['transl'][windows_idxs]).to(torch.float32).reshape(-1,30,3)
                obj_transl = (obj_transl.reshape(-1,30,3) - rel_transl).reshape(-1,3)

                
                obj_rot = torch.tensor(obj_seq_params['global_orient'][windows_idxs]).to(torch.float32)
                obj_rot_rotmat = aa2rotmat(obj_rot).squeeze()
                
                obj_verts = torch.from_numpy(obj_mesh.v).to(torch.float32)
                obj_verts = obj_verts.repeat(obj_transl.shape[0],1,1)
                obj_verts_rot = torch.matmul(obj_verts, obj_rot_rotmat) + obj_transl.reshape(-1,3).unsqueeze(1)
                
                ### visualization 
                # obj_mesh = Mesh(filename=os.path.join(self.root_path, seq_data['object']['object_mesh']))
                # sp_anim = sp_animation()
                # for i in range(30):
                #     sbj_mesh_ = Mesh(v = to_cpu(smplx_vertices).reshape(-1,30,10475,3)[0,i],f = smplx_model.faces, vc = name_to_rgb['green'])
                #     obj_mesh = Mesh(v = to_cpu(obj_verts_rot).reshape(len(windows),30,-1,3)[0,i],f = obj_mesh.f, vc = name_to_rgb['yellow'])
                #     sbj_mesh_2 = Mesh(v = to_cpu(smplx_vertices).reshape(-1,30,10475,3)[-1,i],f = smplx_model.faces, vc = name_to_rgb['green'])
                #     obj_mesh_2 = Mesh(v = to_cpu(obj_verts_rot).reshape(len(windows),30,-1,3)[-1,i],f = obj_mesh.f, vc = name_to_rgb['yellow'])
                    
                #     sp_anim.add_frame([sbj_mesh_, obj_mesh, sbj_mesh_2, obj_mesh_2],['sbj_mesh', 'obj_mesh','sbj_mesh_2', 'obj_mesh_2'])
                # sp_anim.save_animation('grab_check.html')
                
                
                obj_mass = obj_verts_rot.mean(dim=1)
                
                obj_hand_contact = seqs_contact_label[windows_idxs]

                obj_bps = self.bps['obj'].repeat(obj_transl.shape[0],1,1) + obj_transl.reshape(-1, 1, 3)

                
                bps_obj = self.bps_torch.encode(x=obj_verts_rot,
                                                feature_type=['deltas'],
                                                custom_basis=obj_bps)['deltas']
                bps_obj = to_cpu(bps_obj)  
                
                
                lr_joints = smplx_joints.to('cpu')[:,20:22,:]
                lr_joint2omass = lr_joints - obj_mass.reshape(-1,1,3).repeat(1,2,1)
                
                
                ### final injection
                sbj_sid = np.array(int(sid[1:])).repeat(len(windows)).reshape(-1,1).tolist()
                obj_idx = [[obj2idx[obj_name]]]*len(windows)
                
                
                smplx_transl = smplx_transl.reshape(-1,30,3)
                smplx_pose_rotmat = smplx_pose_rotmat.reshape(-1,30,54,3,3)
                smplx_global_rotmat = smplx_global_rotmat.reshape(-1,30,3,3)
                smplx_joints = smplx_joints.reshape(-1,30,127,3)
                lr_joint2omass = lr_joint2omass.reshape(-1,30,2,3)
                
                bps_obj = bps_obj.reshape(-1,30,1024,3)
                obj_rot_rotmat = obj_rot_rotmat.reshape(-1,30,3,3)
                obj_transl = obj_transl.reshape(-1,30,3)
                obj_mass = obj_mass.reshape(-1,30,3)
                obj_hand_contact = obj_hand_contact.reshape(-1,30,2)
                
                with lmdb_env.begin(write=True) as txn:
                    for idxs in range(len(windows)):
                        key = (ds_name + "{:05}".format(ds_id[ds_name])).encode()
                        cache = {'sbj_transl': smplx_transl[idxs,...],
                                'sbj_fullpose_rotmat': smplx_pose_rotmat[idxs,...],
                                'sbj_rot_rotmat': smplx_global_rotmat[idxs,...],
                                'sbj_joints': smplx_joints[idxs,...],
                                'sid': np.array(sbj_sid[idxs]),
                                'lr_joint2omass': lr_joint2omass[idxs,...],
                                
                                'obj_bps_glob': bps_obj[idxs,...],
                                'obj_global_orient_rotmat': obj_rot_rotmat[idxs,...],
                                'obj_transl': obj_transl[idxs,...],
                                'obj_idx': np.array(obj_idx[idxs]),
                                'obj_mass': obj_mass[idxs,...],
                                'obj_hand_contact': obj_hand_contact[idxs,...]
                                }
                        
                        v = pickle.dumps(cache)
                        txn.put(key,v)
                        
                        ds_id[ds_name] += 1
                        
                        ds_welford['pose'] = welford(smplx_pose_rotmat[idxs], ds_welford['pose']['count'], ds_welford['pose']['mean'], ds_welford['pose']['m2'])
                        ds_welford['rot2world'] = welford(smplx_global_rotmat[idxs], ds_welford['rot2world']['count'], ds_welford['rot2world']['mean'], ds_welford['rot2world']['m2'])
                        ds_welford['transl2world'] = welford(smplx_transl[idxs], ds_welford['transl2world']['count'], ds_welford['transl2world']['mean'], ds_welford['transl2world']['m2'])
                        ds_welford['joints'] = welford(smplx_joints[idxs], ds_welford['joints']['count'], ds_welford['joints']['mean'], ds_welford['joints']['m2'])
                        ds_welford['lr_2omass'] = welford(lr_joint2omass[idxs], ds_welford['lr_2omass']['count'], ds_welford['lr_2omass']['mean'], ds_welford['lr_2omass']['m2'])
                print(ds_id)
        
        print("Calculating dataset statistical result...")
        print(ds_welford['pose']['count'])
        ds_info = {'smplx_info':{}}
        ds_info['smplx_info']['pose_mean'] = ds_welford['pose']['mean']
        ds_info['smplx_info']['pose_std'] = np.sqrt(ds_welford['pose']['m2']/(ds_welford['pose']['count']-1))
        ds_info['smplx_info']['rot2world_mean'] = ds_welford['rot2world']['mean']
        ds_info['smplx_info']['rot2world_std'] = np.sqrt(ds_welford['rot2world']['m2']/(ds_welford['rot2world']['count']-1))
        ds_info['smplx_info']['transl2world_mean'] = ds_welford['transl2world']['mean']
        ds_info['smplx_info']['transl2world_std'] = np.sqrt(ds_welford['transl2world']['m2']/(ds_welford['transl2world']['count']-1))
        ds_info['smplx_info']['joints_mean'] = ds_welford['joints']['mean']
        ds_info['smplx_info']['joints_std'] = np.sqrt(ds_welford['joints']['m2']/(ds_welford['joints']['count']-1))
        ds_info['smplx_info']['lr_2omass_mean'] = ds_welford['lr_2omass']['mean']
        ds_info['smplx_info']['lr_2omass_std'] = np.sqrt(ds_welford['lr_2omass']['m2']/(ds_welford['lr_2omass']['count']-1))
        ds_info['ds_len'] = ds_id
        
        out_data = [ds_info]
        out_data_name = ['ds_info']
        import _pickle as pickle
        for idx, _ in enumerate(out_data):
            data_name = out_data_name[idx]
            outfname = makepath(os.path.join(self.out_path, '%s.npy' % data_name), isfile=True)
            pickle.dump(out_data[idx], open(outfname, 'wb'), protocol=4)
            
        d_outfname = os.path.join(self.out_path, 'idx2obj.pickle')
        pickle.dump(idx2obj, open(d_outfname, 'wb'))
            

    
    def sum_contact_frames(self, contact_label_body):
        
        seq_len = contact_label_body.shape[0]
        
        bool_list_l = np.zeros([seq_len])
        bool_list_r = np.zeros([seq_len])
        
        left_contact = np.sum(contact_label_body[:, self.smplx_mano_left_verts_bool], axis=1) 
        right_contact = np.sum(contact_label_body[:, self.smplx_mano_right_verts_bool], axis=1)
        
        bool_list_l[left_contact > 0] = 1
        bool_list_r[right_contact > 0] = 1
        
        bool_list_l = torch.from_numpy(bool_list_l).to(torch.bool).unsqueeze(1)
        bool_list_r = torch.from_numpy(bool_list_r).to(torch.bool).unsqueeze(1)
        
        contact_label = torch.cat((bool_list_l,bool_list_r),dim=1)
        
        contact_info = torch.sum(contact_label,dim=1).tolist()
        first_nonzero_index = next((i for i, x in enumerate(contact_info) if x != 0), None)
        last_nonzero_index = len(contact_info) - next((i for i, x in enumerate(reversed(contact_info)) if x != 0), None) - 1

        return (first_nonzero_index, last_nonzero_index), contact_label
        
    def load_sbj_verts(self, sbj_id, seq_data):
        
        mesh_path = os.path.join(self.root_path, seq_data.body.vtemp)
        betas_path = mesh_path.replace('.ply', '_betas.npy')
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]['vtemp']
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).v)
            sbj_betas = np.load(betas_path)
            self.sbj_info[sbj_id] = {'vtemp': sbj_vtemp,
                                     'gender': seq_data.gender,
                                     'betas': sbj_betas}
        return sbj_vtemp
    
    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=2048):

        mesh_path = os.path.join(self.root_path, seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            np.random.seed(100)
            obj_mesh = Mesh(filename=mesh_path)
            verts_obj = np.array(obj_mesh.v)
            faces_obj = np.array(obj_mesh.f)

            if verts_obj.shape[0] > n_verts_sample:
                verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            verts_sampled = verts_obj[verts_sample_id]
            self.obj_info[obj_name] = {'verts': verts_obj,
                                       'faces': faces_obj,
                                       'verts_sample_id': verts_sample_id,
                                       'verts_sample': verts_sampled,
                                       'obj_mesh_file': mesh_path}

        return self.obj_info[obj_name]
    
    def prepare_params(self, params, rel_trans = None, dtype = np.float32):
        n_params = {k: v.astype(dtype)  for k, v in params.items()}
        if rel_trans is not None:
            n_params['transl'] -= rel_trans
        return n_params
    



def tensor_normalization(source_tensor, tensor_MAX, tensor_MIN):
    # to [0,1]
    tensor_norm = torch.div((source_tensor - tensor_MIN),(tensor_MAX - tensor_MIN))
    # to [-1,1]
    tensor_norm = tensor_norm * 2. - 1.
    
    return tensor_norm
    
def tensor_denormalization(source_tensor, tensor_MAX, tensor_MIN):
    tensor_denorm = source_tensor + 1.
    tensor_denorm /= 2.
    tensor_denorm = tensor_denorm * (tensor_MAX - tensor_MIN) + tensor_MIN
    return tensor_denorm

def tensor_standardization(source_tensor, mean, std):
    std = torch.where(std == 0, torch.tensor(1e-8), std)
    return (source_tensor - mean) / std

def tensor_destandardization(source_tensor, mean, std):
    return source_tensor * std + mean

def get_ground(cage_size = 7, grnd_size = 5, axis_size = 1):
    ax_v = np.array([[0., 0., 0.],
                     [1.0, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
    ax_e = [(0, 1), (0, 2), (0, 3)]
    from psbody.mesh.lines import Lines
    axis_l = Lines(axis_size*ax_v, ax_e, vc=np.eye(4)[:, 1:])

    base_width = 0.2
    width = base_width * grnd_size

    g_points = np.array([[-width, -width, -0.03],
                         [width, width, -0.03],
                         [width, -width, -0.03],
                         [-width, width, -0.03]])
    
    g_faces = np.array([[2,1,0], [1,3,0]])
    grnd_mesh = Mesh(v=g_points, f=g_faces, vc=name_to_rgb['gray'])

    cage_points = np.array([[-.2, .0, -.2],
                            [.2, .2, .2],
                            [.2, 0., 0.2],
                            [-.2, .2, -.2]])
    cage = [Mesh(v=cage_size * cage_points, f=[], vc=name_to_rgb['black'])]
    return grnd_mesh, cage, axis_l

def glob2rel(motion_sbj, motion_obj, R,root_offset, rel_trans=None):

    fpose_sbj_rotmat = aa2rotmat(motion_sbj['fullpose'])
    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
    fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

    trans_sbj_rel = rotate((motion_sbj['transl'] + root_offset), R) - root_offset
    trans_obj_rel = rotate(motion_obj['transl'], R)

    global_orient_obj_rotmat = aa2rotmat(motion_obj['global_orient'])
    global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

    if rel_trans is None:
        rel_trans = trans_sbj_rel.clone()
        rel_trans[:,1] -= rel_trans[:,1]

    motion_sbj['transl'] = to_tensor(trans_sbj_rel)
    motion_sbj['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
    motion_sbj['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
    motion_sbj['fullpose'][:, :3] = motion_sbj['global_orient']
    motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

    motion_obj['transl'] = to_tensor(trans_obj_rel)
    motion_obj['global_orient'] = rotmat2aa(to_tensor(global_orient_obj_rel).squeeze()).squeeze()
    motion_obj['global_orient_rotmat'] = to_tensor(global_orient_obj_rel)

    return motion_sbj, motion_obj, rel_trans

def dict2np(item):
    out = {}
    
    for k, v in item.items():   
        if v == []:
            continue   
        out[k] = np.array(v)
    return out

def welford(indata, count, mean, m2):
    if count is None:
        return {'count': 1, 'mean': indata, 'm2': indata*indata}
    count += 1
    delta = indata - mean
    mean += delta / count
    
    m2 += delta * (indata - mean)
    return {'count': count, 'mean': mean, 'm2': m2}

if __name__ == '__main__':

    ArcticDataSet()