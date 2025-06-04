import sys
sys.path.append('')
sys.path.append('..')
import os

import glob
import numpy as np
import torch
from torch.utils import data

import pickle 
import lmdb
import trimesh


class LoadData(data.Dataset):
    
    def __init__(self, cfg=None, split_name='train') -> None:
        
        super().__init__()
        
        self.cwd = os.path.dirname(sys.argv[0])
        self.split_name = split_name
        self.ds_dir = cfg.dataset_dir
        self.cfg = cfg
        self.lmdb_env = lmdb.open(os.path.join(self.ds_dir,split_name), readonly=True, lock=False)
        
        self.sbj_info = self.load_sbj_vtemp()
    
    def load_sbj_vtemp(self):
        sid_gender = {1:'male',2:'male',3:'female',4:'female',5:'female',
                      6:'female',7:'female',8:'male',9:'male',10:'male'}
        all_sbj = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
        self.all_seqs = glob.glob(os.path.join(self.ds_dir ,'subject_meshes/*/*.ply'))
        sbj_info = {}
        for sbj in all_sbj:
            sbj_mesh_path = os.path.join(self.ds_dir ,'subject_meshes/{}/{}.ply'.format(sid_gender[int(sbj[1:])], sbj))
            sbj_mesh = trimesh.load(sbj_mesh_path, process=False)
            sbj_betas = os.path.join(self.ds_dir ,'subject_meshes/{}/{}_betas.npy'.format(sid_gender[int(sbj[1:])], sbj))
            sbj_betas = np.load(sbj_betas)
            sbj_info[int(sbj[1:])] = {'betas':sbj_betas, 'v_template': sbj_mesh.vertices, 'gender':sid_gender[int(sbj[1:])]}
        return sbj_info
    
 
    def load_idx_seq(self, seq_idx):
       
        out_dict = {}
        
        seq_0 = self.load_idx(seq_idx[0])

        keys = seq_0.keys()
        out_dict = {k:[] for k in keys}
        
        for i in range(len(seq_idx)):
            frame = self.load_idx(seq_idx[i])
            for k,v in frame.items():
                out_dict[k].append(v)
        for k in keys:
            out_dict[k] = torch.stack(out_dict[k])
        
        return out_dict
    
    def __len__(self):

        return self.lmdb_env.stat()['entries']
    
    
    def __getitem__(self, idx):
        out = {}
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:05}".format(idx)
            sample = txn.get(key.encode())
            sample = pickle.loads(sample)

            ### subject vtemplate, gender, betas
            sid = int(sample['sid'][0])
            v_template = self.sbj_info[sid]['v_template']
            betas = self.sbj_info[sid]['betas']
            gender = self.sbj_info[sid]['gender']
            gender = 1 if gender == 'male' else 0
        
            out['sbj_smplx_transl'] = torch.tensor(sample['sbj_transl'])
            out['sbj_fullpose'] = torch.tensor(sample['sbj_fullpose'])
            out['sbj_smplx_joints'] = torch.tensor(sample['sbj_joints'])
            out['sbj_global_orient'] = torch.tensor(sample['sbj_global_orient'])
            out['sbj_lr2omas'] = torch.tensor(sample['lr_joint2omass'])
            
            out['gender'] = torch.tensor(np.array(gender).reshape(1,-1).repeat(30,axis=0))
            out['smplx_v_temp'] = torch.tensor(np.array(v_template).reshape(1,10475,3).repeat(30,axis=0))
            out['beta'] = torch.tensor(betas.reshape(1,-1).repeat(30,0))
            
            out['obj_bps_glob'] = torch.tensor(sample['obj_bps_glob'])
            out['obj_transl'] = torch.tensor(sample['obj_transl'])
            out['obj_global_orient'] = torch.tensor(sample['obj_global_orient'])
            out['obj_idx'] = torch.tensor(sample['obj_idx'])
            out['obj_mass'] = torch.tensor(sample['obj_mass'])
            out['contact_label'] = torch.tensor(sample['obj_hand_contact'])

        return out
    
    
def build_dataloader(dataset: torch.utils.data.Dataset,
                     split: str = 'train',
                     cfg = None):

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  =   cfg.batch_size if split != 'test' else 1, 
        num_workers =   1 if split != 'test' else 1,
        drop_last   =   False,
        shuffle     =   False,
    )
    return data_loader
        