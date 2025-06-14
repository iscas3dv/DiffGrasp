from typing import Tuple, Optional, Any, Union, List
from loguru import logger
from copy import deepcopy

from dataclasses import dataclass
from omegaconf import OmegaConf

import os, sys
cdir = os.path.dirname(sys.argv[0])


@dataclass
class DatasetConfig:
    batch_size: int = 256
    use_equal_sampling: bool = True
    use_packed: bool = True
    use_face_contour: bool = True

    dataset_dir: str = f''
    objects_dir: str = ''
    grab_path: str = ''

    fps: int = 30
    past_frames: int =  10
    future_pred: int = 10
    chunk_size: int = 21

    model_path: str =  ''
    dataset_dir: str = '/data/heqiang/GRAB_diffgrasp_wostd'


@dataclass
class DiffusionNet:
    hidden_size: int = 300
    pose_dim: int = 99
    diff_hidden_dim: int = 512
    block_depth: int = 8
    model: str = 'pose_diffusion'
    null_cond_prob: float = 0.1
    motion_resampling_framerate: int = 15
    n_poses: int = 1
    bps_size: int = 0
    subdivision_stride: int = 10
    loader_workers: int = 4
    classifier_free: bool = True
    
    gen_length: int = 40
    # mean_dir_vec: List[float] = [-0.0737964, -0.9968923, -0.1082858,  0.9111595,  0.2399522, -0.102547 , -0.8936886,  0.3131501, -0.1039348,  0.2093927, 0.958293 ,  0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763, -0.2495191, -0.614666 , -0.3877234,  0.005006 , -0.5301695, -0.5098616,  0.2257808,  0.0053111, -0.2393621, -0.1022204, -0.6583039, -0.4992898,  0.1228059, -0.3292085, -0.4753748,  0.2132857,  0.1742853, -0.2062069,  0.2305175, -0.5897119, -0.5452555,  0.1303197, -0.2181693, -0.5221036, 0.1211322,  0.1337591, -0.2164441,  0.0743345, -0.6464546, -0.5284583,  0.0457585, -0.319634 , -0.5074904,  0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, -0.2417618, -0.3054556, -0.3556116, -0.281753 , -0.5164358, -0.3064435,  0.9284261, -0.067134 ,  0.2764367,  0.006997 , -0.7365526,  0.2421269, -0.225798 , -0.6387642,  0.3788997, 0.0283412, -0.5451686,  0.5753376,  0.1935219,  0.0632555, 0.2122412, -0.0624179, -0.6755542,  0.5212831,  0.1043523, -0.345288 ,  0.5443628,  0.128029 ,  0.2073687,  0.2197118, 0.2821399, -0.580695 ,  0.573988 ,  0.0786667, -0.2133071, 0.5532452, -0.0006157,  0.1598754,  0.2093099,  0.124119, -0.6504359,  0.5465003,  0.0114155, -0.3203954,  0.5512083, 0.0489287,  0.1676814,  0.4190787, -0.4018607, -0.3912126, 0.4841548, -0.2668508, -0.3557675,  0.3416916, -0.2419564, -0.5509825,  0.0485515, -0.6343101, -0.6817347, -0.4705639, -0.6380668,  0.4641643,  0.4540192, -0.6486361,  0.4604001, -0.3256226,  0.1883097,  0.8057457,  0.3257385,  0.1292366, 0.815372]
    # mean_pose: List[float] = [-0.0046788, -0.5397806,  0.007695 , -0.0171913, -0.7060388,-0.0107034,  0.1550734, -0.6823077, -0.0303645, -0.1514748,   -0.6819547, -0.0268262,  0.2094328, -0.469447 , -0.0096073,   -0.2318253, -0.4680838, -0.0444074,  0.1667382, -0.4643363,   -0.1895118, -0.1648597, -0.4552845, -0.2159728,  0.1387546,   -0.4859474, -0.2506667,  0.1263615, -0.4856088, -0.2675801,   0.1149031, -0.4804542, -0.267329 ,  0.1414847, -0.4727709,   -0.2583424,  0.1262482, -0.4686185, -0.2682536,  0.1150217,   -0.4633611, -0.2640182,  0.1475897, -0.4415648, -0.2438853,   0.1367996, -0.4383164, -0.248248 ,  0.1267222, -0.435534 ,   -0.2455436,  0.1455485, -0.4557491, -0.2521977,  0.1305471,   -0.4535603, -0.2611591,  0.1184687, -0.4495366, -0.257798 ,   0.1451682, -0.4802511, -0.2081622,  0.1301337, -0.4865308,   -0.2175783,  0.1208341, -0.4932623, -0.2311025, -0.1409241,-0.4742868, -0.2795303, -0.1287992, -0.4724431, -0.2963172,-0.1159225, -0.4676439, -0.2948754, -0.1427748, -0.4589126,-0.2861245, -0.126862 , -0.4547355, -0.2962466, -0.1140265,-0.451308 , -0.2913815, -0.1447202, -0.4260471, -0.2697673,-0.1333492, -0.4239912, -0.2738043, -0.1226859, -0.4238346,-0.2706725, -0.1446909, -0.440342 , -0.2789209, -0.1291436,-0.4391063, -0.2876539, -0.1160435, -0.4376317, -0.2836147,-0.1441438, -0.4729031, -0.2355619, -0.1293268, -0.4793807,-0.2468831, -0.1204146, -0.4847246, -0.2613876, -0.0056085,-0.9224338, -0.1677302, -0.0352157, -0.963936 , -0.1388849,0.0236298, -0.9650772, -0.1385154, -0.0697098, -0.9514691,-0.055632 ,  0.0568838, -0.9565502, -0.0567985]

@dataclass
class MDM:
    
    use_pointnet2: int = 1
    sample_rate: int = 1
    smpl_dim: int = 333
    num_obj_points: int = 0
    
    latent_dim: int  = 256
    embedding_dim: int = 512
    num_heads: int = 8
    ff_size: int = 1024
    activation: str = 'gelu'
    dropout: float = 0
    num_layers: int = 4
    latent_usage: str = 'memory'
    template_type: str = 'zero'
    star_graph: bool = False
    
    lr: float = 3e-4
    l2_norm: float = 0
    robust_kl: int = 1
    weight_template: float = 0.1
    weight_kl: float = 1e-2
    weight_contact: float = 0
    weight_dist: float = 1
    weight_penetration: float = 0
    
    weight_smplx_rot: float = 1
    weight_smplx_nonrot: float = 0.2
    weight_obj_rot: float = 0.1
    weight_obj_nonrot: float = 0.2
    weight_past: float = 1
    weight_jtr: float = 0.1
    weight_jtr_v: float = 500
    weight_v: float = 0.2
    
    use_contact: int = 0
    use_annealing: int = 0
    
    future_len: int = 30
    past_len: int = 0
    
    diffusion_steps: int = 1000
    cond_mask_prob: float = 0
    sigma_small: bool = True
    noise_schedule: str = 'cosine'
    diverse_samples: int = 10
    
    

@dataclass
class Config:
    mode: str = ''
    is_continue: bool = False
    multi_vis: bool = False

    description: str = ''
    num_gpus: int = 1
    use_cuda: bool = True
    is_training: bool = True
    logger_level: str = 'info'

    seed: int = 3407
    n_epochs: int = 300

    output_folder: str = f'results'
    work_dir: str = f''
    results_base_dir: str = ''
    expr_ID: str = 'test'

    best_model: Optional[str] = None

    datasets: DatasetConfig = DatasetConfig()
    diffusion_net: DiffusionNet = DiffusionNet()
    mdm: MDM = MDM()

conf = OmegaConf.structured(Config)