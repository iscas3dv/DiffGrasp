import sys
sys.path.append('')
sys.path.append('..')
import numpy as np
import torch
import os, shutil, glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm

from tools.utils import makepath
from tools.utils import parse_npz
from tools.utils import to_cpu
from tools.utils import aa2rotmat
from models.model_utils import parms_6D2full

from bps_torch.bps import bps_torch
import trimesh

from smplx import SMPLXLayer 

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
        self.root_path = '/data/to/GRAB'
        self.out_path = '/data/processed_GRAB'
        
        self.seq_path = os.path.join(self.root_path, 'grab')
        self.tools_path = os.path.join(self.root_path, 'tools')

        
        self.total_lens = 0 
        self.all_seqs = []
        self.all_seqs = glob.glob(os.path.join(self.seq_path ,'*/*.npz'))
        
        import lmdb
        lmdb_path = self.out_path
        lmdb_path_train = makepath(os.path.join(lmdb_path,'train'))
        lmdb_env_train = lmdb.open(lmdb_path_train, map_size=987842478080)
        lmdb_path_val = makepath(os.path.join(lmdb_path,'val'))
        lmdb_env_val = lmdb.open(lmdb_path_val, map_size=987842478080)  
        lmdb_path_test = makepath(os.path.join(lmdb_path,'test'))
        lmdb_env_test = lmdb.open(lmdb_path_test, map_size=987842478080)  
        
        self.all_objs = []
        for i in range(len(self.all_seqs)):
            obj_t = self.all_seqs[i].split('/')[-1].split('_')[0]
            self.all_objs.append(obj_t)

        ### all object for training
        # ## without 5 object (apple, toothbrush, train, cubelarge, phone) for unknown object testing
        self.object_list = ['cylinderlarge', 'mug', 'elephant', 'stanfordbunny', 'airplane', 'alarmclock', 'banana', 'body', 'bowl', 'cubesmall', 'cup', 'cubemedium',
             'eyeglasses', 'flashlight', 'flute', 'gamecontroller', 'hammer', 'headphones', 'knife', 'lightbulb', 'mouse', 'piggybank', 'pyramidlarge', 'pyramidsmall', 'pyramidmedium',
             'duck', 'scissors', 'spherelarge', 'spheresmall', 'stamp', 'stapler', 'teapot', 'toruslarge', 'torussmall', 'watch', 'cylindersmall', 'waterbottle', 'torusmedium',
             'cylindermedium', 'spheremedium', 'wristwatch', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste','doorknob', 'hand']

        # Move object meshes to a new folder
        for obj in self.object_list:
            obj_mesh_path = makepath(os.path.join(self.out_path, 'obj_meshes', '%s.obj' % obj), isfile=True)
            obj_mesh = trimesh.load(os.path.join(self.tools_path, 'object_meshes/processed_object_meshes', obj+'.ply'), process=False)
            obj_mesh.export(obj_mesh_path)

        
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
            v_templates = trimesh.load(v_templates, process=False)

            gender = 1 if sub_gender == 'male' else 0
            beta = os.path.join(self.tools_path, 'subject_meshes', sub_gender, sid+'_betas.npy')
            beta = np.load(beta)
            
            obj_seq_params = seq_data['object']['params']
            obj_mesh = trimesh.load(os.path.join(self.root_path, seq_data['object']['object_mesh']), process=False)
            
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
                lmdb_env = lmdb_env_train
            elif sid in self.val_sbj:
                ds_name = 'val'
                lmdb_env = lmdb_env_val
            else:
                ds_name = 'test'
                lmdb_env = lmdb_env_test
            
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
                    self.male_model.v_template = torch.tensor(v_templates.vertices).to(torch.float32)
                    smplx_model = self.male_model.to(device)
                    
                else:
                    self.female_model.v_template = torch.tensor(v_templates.vertices).to(torch.float32)
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
                smplx_joints = output_gt.joints.cpu()

                ### obj processing
                obj_transl = torch.tensor(obj_seq_params['transl'][windows_idxs]).to(torch.float32).reshape(-1,30,3)
                obj_transl = (obj_transl.reshape(-1,30,3) - rel_transl).reshape(-1,3)

                
                obj_rot = torch.tensor(obj_seq_params['global_orient'][windows_idxs]).to(torch.float32)
                obj_rot_rotmat = aa2rotmat(obj_rot).squeeze()
                
                obj_verts = torch.from_numpy(obj_mesh.vertices).to(torch.float32)
                obj_verts = obj_verts.repeat(obj_transl.shape[0],1,1)
                obj_verts_rot = torch.matmul(obj_verts, obj_rot_rotmat) + obj_transl.reshape(-1,3).unsqueeze(1)
                
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
                
                
                smplx_transl = smplx_transl.reshape(-1,30,3).contiguous().numpy()
                smplx_pose_rotmat = smplx_pose_rotmat.reshape(-1,30,54,3,3).contiguous().numpy()
                smplx_global_rotmat = smplx_global_rotmat.reshape(-1,30,3,3).contiguous().numpy()
                smplx_joints = smplx_joints.reshape(-1,30,127,3)
                lr_joint2omass = lr_joint2omass.reshape(-1,30,2,3)
                
                bps_obj = bps_obj.reshape(-1,30,1024,3)
                obj_rot_rotmat = obj_rot_rotmat.reshape(-1,30,3,3).contiguous().numpy()
                obj_transl = obj_transl.reshape(-1,30,3).contiguous().numpy()
                obj_mass = obj_mass.reshape(-1,30,3).contiguous().numpy()
                obj_hand_contact = obj_hand_contact.reshape(-1,30,2).contiguous().numpy()
                
                with lmdb_env.begin(write=True) as txn:
                    for idxs in range(len(windows)):
                        key = ("{:05}".format(ds_id[ds_name])).encode()
                        cache = {'sbj_transl': smplx_transl[idxs,...].tolist(),
                                'sbj_fullpose': smplx_pose_rotmat[idxs,...].tolist(),
                                'sbj_global_orient': smplx_global_rotmat[idxs,...].tolist(),
                                'sbj_joints': smplx_joints[idxs,...].tolist(),
                                'sid': np.array(sbj_sid[idxs]).tolist(),
                                'lr_joint2omass': lr_joint2omass[idxs,...].tolist(),
                                
                                'obj_bps_glob': bps_obj[idxs,...].tolist(),
                                'obj_global_orient': obj_rot_rotmat[idxs,...].tolist(),
                                'obj_transl': obj_transl[idxs,...].tolist(),
                                'obj_idx': np.array(obj_idx[idxs]).tolist(),
                                'obj_mass': obj_mass[idxs,...].tolist(),
                                'obj_hand_contact': obj_hand_contact[idxs,...].tolist()
                                }
                        
                        v = pickle.dumps(cache)
                        txn.put(key,v)
                        
                        ds_id[ds_name] += 1
                print(ds_id)
            
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
        

def dict2np(item):
    out = {}
    
    for k, v in item.items():   
        if v == []:
            continue   
        out[k] = np.array(v)
    return out

if __name__ == '__main__':

    ArcticDataSet()