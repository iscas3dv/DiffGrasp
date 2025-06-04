import torch
import torch.nn as nn
import torch.nn.functional as F
from bmdm_models.layers import FeaturePositionalEncoding,Feature4PositionalEncoding, FeatureUpDim, CL_PositionalEncoding, PositionalEncoding, TimestepEmbedder, TransformerEncoder, TransformerDecoder, ConEncoder
from bmdm_models.sublayers import TransformerDecoderLayerQaN, TransformerEncoderLayerQaN

class MDM(nn.Module):
    def __init__(self, args):
        super(MDM, self).__init__()
        self.args = args
        num_channels = args.embedding_dim
        decoder_in = 339
        self.bodyEmbedding = nn.Linear(decoder_in, num_channels)
        
        # self.objEmbedding = nn.Linear(9, num_channels)
        self.conEmbedding = ConEncoder(256+6+3+3+1+10+2, num_channels)
        self.CL_conEmbedding = ConEncoder(256+6+3+3+1+10, num_channels)
        self.CLalignEmbedding = ConEncoder(256 * 3, num_channels)
        self.sentenceAlignEmbedding = ConEncoder(256 * 4, num_channels)
        self.PositionalEmbedding = PositionalEncoding(d_model=num_channels, dropout=args.dropout)
        self.ObjparamsEmbedding = FeatureUpDim(in_dim=6+3+3, out_dim=256)
        self.SbjparamsEmbedding = FeatureUpDim(in_dim=10+1, out_dim=256)
        self.CLparamsEmbedding = FeatureUpDim(in_dim=2, out_dim=256)
        self.FeaturePositionalEncoding = FeaturePositionalEncoding(256, dropout=args.dropout)
        self.Feature4PositionalEncoding = Feature4PositionalEncoding(256, dropout=args.dropout)
        self.CL_PositionalEmbedding = CL_PositionalEncoding(d_model=num_channels, dropout=args.dropout)
        
        self.embedTimeStep = TimestepEmbedder(num_channels, self.PositionalEmbedding)
        self.objPooling = torch.nn.MaxPool1d(1)
        from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
        seqTransEncoderLayer1 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer2 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer3 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer4 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer5 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer6 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer7 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer8 = TransformerEncoderLayer(d_model=num_channels,
                                                            nhead=self.args.num_heads,
                                                            dim_feedforward=self.args.ff_size,
                                                            dropout=self.args.dropout,
                                                            activation=self.args.activation,
                                                            batch_first=False)
        seqTransEncoderLayer = nn.ModuleList([seqTransEncoderLayer1, seqTransEncoderLayer2, seqTransEncoderLayer3, seqTransEncoderLayer4,
                                              seqTransEncoderLayer5, seqTransEncoderLayer6, seqTransEncoderLayer7, seqTransEncoderLayer8])
        self.encoder = TransformerEncoder(seqTransEncoderLayer)

        if self.args.latent_usage == 'memory':
            seqTransDecoderLayer1 = TransformerDecoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer2 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer3 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer4 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer5 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer6 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer7 = TransformerDecoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer8 = TransformerDecoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer = nn.ModuleList([seqTransDecoderLayer1, seqTransDecoderLayer2, seqTransDecoderLayer3, seqTransDecoderLayer4,
                                                  seqTransDecoderLayer5, seqTransDecoderLayer6, seqTransDecoderLayer7, seqTransDecoderLayer8])
            self.decoder = TransformerDecoder(seqTransDecoderLayer)
        else:
            seqTransDecoderLayer1 = TransformerEncoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer2 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer3 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer4 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer5 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer6 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer7 = TransformerEncoderLayerQaN(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer8 = TransformerEncoderLayer(d_model=num_channels,
                                                              nhead=self.args.num_heads,
                                                              dim_feedforward=self.args.ff_size,
                                                              dropout=self.args.dropout,
                                                              activation=self.args.activation,
                                                              batch_first=False)
            seqTransDecoderLayer = nn.ModuleList([seqTransDecoderLayer1, seqTransDecoderLayer2, seqTransDecoderLayer3, seqTransDecoderLayer4,
                                                  seqTransDecoderLayer5, seqTransDecoderLayer6, seqTransDecoderLayer7, seqTransDecoderLayer8])
            self.decoder = TransformerEncoder(seqTransDecoderLayer)

        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3,out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512,out_features=256),)
        
        self.sentence_encoder = nn.Sequential(
            nn.Linear(in_features=512,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=256),)


        self.finalLinear = nn.Linear(num_channels, args.smpl_dim+9)
        self.bodyFinalLinear = nn.Linear(num_channels, decoder_in)
        self.objFinalLinear = nn.Linear(num_channels, 9)
        self.bodyFutureEmbedding = nn.Parameter(torch.FloatTensor(args.future_len, 1, num_channels)) 
        self.bodyFutureEmbedding.data.uniform_(-1,1)
        self.objFutureEmbedding = nn.Parameter(torch.FloatTensor(args.future_len, 1, num_channels)) 
        self.objFutureEmbedding.data.uniform_(-1,1)

    def mask_cond(self, cond, force_mask=False):
        
        t, BS, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.args.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(BS, device=cond.device) * self.args.cond_mask_prob).view(1, BS, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    
    def _get_cond_embeddings_align(self, x, device, is_rel=False):
        
        BS = x['obj_transl'].shape[0]
        T = x['obj_transl'].shape[1]
        
        condition_obj = {}
        condition_obj['obj_transl'] = x['obj_transl'] # 3
        condition_obj['obj_global_orient'] = x['obj_global_orient'][:,:,:2,:] # 6
        condition_obj['obj_mass'] = x['obj_mass']
        condition_obj = torch.cat([v.reshape(BS, T, -1).to(device) for v in condition_obj.values()], dim=2)
        condition_obj = self.ObjparamsEmbedding(condition_obj)
        
        condition_sbj = {}
        condition_sbj['gender'] = x['gender'] # 1
        condition_sbj['beta'] = x['beta'] # 10
        condition_sbj = torch.cat([v.reshape(BS, T,-1).to(device) for v in condition_sbj.values()], dim=2)
        condition_sbj = self.SbjparamsEmbedding(condition_sbj)
        
        condition_bps = self.bps_encoder(x['obj_bps_glob'].reshape(BS, T, -1).to(device))

        condition = torch.cat([condition_obj, condition_bps, condition_sbj], dim=2)
         ### feature PE
        condition = self.FeaturePositionalEncoding(condition)
        
        
        condition = condition.permute(1,0,2).contiguous()
        
        condition = self.CLalignEmbedding(condition)
        # torch.Size([T, BS, 512])
        
        condition = self.PositionalEmbedding(condition)

        condition = self.encoder(condition)
        # torch.Size([T, BS, 512])
        return condition
    
        
    def _decode(self, x, time_embedding, y=None):
        body = self.bodyEmbedding(x)

        decoder_input = body + time_embedding

        decoder_input = self.PositionalEmbedding(decoder_input)
        
        decoder_output = self.decoder(tgt=decoder_input, memory=y)

        body = self.bodyFinalLinear(decoder_output)
        
        return body

    def forward(self, x, timesteps, y=None):

        # BS, 512
        time_embedding = self.embedTimeStep(timesteps).squeeze(1)

        x = x.squeeze(1).permute(2,0,1).contiguous()
        ### torch.Size([T, BS, 339])
        
        if y is not None:
            y = self.mask_cond(y['cond'])
        x_0 = self._decode(x, time_embedding, y)
        x_0 = x_0.permute(1,2,0).unsqueeze(1).contiguous()
        # torch.Size([BS, 1, 339, T])
        return x_0

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  
    steps = args.diffusion_steps
    scale_beta = 1.  
    timestep_respacing = ''  
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.weight_v,
    )

def create_model_and_diffusion(args):
    model = MDM(args)
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion