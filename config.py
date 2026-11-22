import os
from easydict import EasyDict as edict
import time
from pathlib import Path

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 42 # random seed,  for reproduction 3035
__C.DATASET = 'MOC_RS' # dataset selection: RGBT, DRONERGBT, GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD



#是否使用NIR
__C.MM = False

__C.NET = 'SAM2MCC'  # Use 'MCC' for the original Swin baseline


__C.PRE_BACKBONE_WEIGHT = Path(__file__).resolve().parent / "pretrained_models" / "sam2.1_hiera_large.pt"
__C.DATA_PATH = os.environ.get("NWPU_MOC_ROOT", str(Path(__file__).resolve().parent / "NWPU-MOC"))
__C.MAX_STEPS = None
__C.VAL_SAMPLES = None

__C.SAM2_ARCH = edict(
    backbone=dict(
        hiera=dict(
            embed_dim=144,
            num_heads=2,
            stages=(2, 6, 36, 4),
            global_att_blocks=(23, 33, 43),
            window_pos_embed_bkg_spatial_size=(7, 7),
            window_spec=(8, 4, 16, 8),
        ),
        channels=(144, 288, 576, 1152),
        input_multiple=32,
    ),
    fpn=dict(width=128),
    decode_head=dict(width=64),
    nir=dict(channels=1),
    mask=dict(classes=2),
)


__C.POS_EMBEDDING = False

__C.PRE_GCC = False # use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = '' # path to model

__C.RESUME = False # contine training
__C.RESUME_PATH = '' # 

__C.BACKBONE_FREEZE = False

__C.GPU_ID = [2] # sigle gpu: [0], [1] ...; multi gpus: [0,1]

__C.TRAIN_SIZE = (512,512) 

__C.TRAIN_BATCH_SIZE = 4

__C.BASE_LR = 5e-5  # FPN and counting layers
__C.CONV_LR = 1e-6  # Image backbone
__C.WEIGHT_DECAY = 1e-4

__C.LOSS_FUNCTION = 'Mse_loss' # 'Bay_loss' or 'Mse_loss' or 'Bmc_loss' or 'Mix_loss' or 'Trans_loss' or 'Cos_loss' 

__C.MODEL_ARCH = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18],
        # depths=[4, 4, 4],
        num_heads=[4, 8, 16],
        # num_heads=[8, 8, 16],
        out_indices=(0, 1, 2),
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[128, 256, 512]
    ))

__C.PRE_WEIGHTS = 'pretrained_models/swin_base_patch4_window7_224.pth'


__C.MAX_EPOCH = 200
__C.alpha = 1

# print 
__C.PRINT_FREQ = 20

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_loss_' + __C.LOSS_FUNCTION \
             + '_alpha_' + str(__C.alpha) \
             + '_base_lr_' + str(__C.BASE_LR) \
             + '_NIR=' + str(__C.MM)
             

             
__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------

__C.VAL_STAGE = [0,50,100,200]
__C.VAL_FREQ = [10,4,4,4]

__C.CP_FREQ = 10

#================================================================================
