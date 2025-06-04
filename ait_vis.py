from aitviewer.remote.viewer import RemoteViewer
from aitviewer.remote.renderables.meshes import RemoteMeshes

import os
import numpy as np


expr_id = 'DiffGrasp'
ds_name = 'train'
epoch_id = 'epoch_'+str(0)
it = 400

npy_path = os.path.join('./work_dir',expr_id,'visualization',epoch_id,ds_name,str(it)+'_0.npy')


animation_dict = np.load(npy_path, allow_pickle=True).item()
mesh_obj = animation_dict['obj']
mesh_gt = animation_dict['gt']
mesh_pred = animation_dict['pred']
    
v_obj_array = np.array(mesh_obj['verts'])
v_gt_array = np.array(mesh_gt['verts'])
v_pred_array = np.array(mesh_pred['verts'])

v = RemoteViewer("10.0.0.188")

rm_gt = RemoteMeshes(v, v_gt_array, mesh_gt['faces'],
                     color=(0.7, 0.2, 0.2, 1.0), name="gt")
rm_pred = RemoteMeshes(v, v_pred_array, mesh_pred['faces'], 
                       color=(0.7, 0.2, 0.2, 1.0), name="pred")
rm_obj = RemoteMeshes(v, v_obj_array, mesh_obj['faces'], 
                      color=(200 / 255, 200 / 255, .0, 1.0), name="object")



