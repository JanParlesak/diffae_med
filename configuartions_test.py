
# Get data 
# Convert to lmdb files 
# resize data


######### TRAIN DIFFAE ######################

#change path in dataset file and write own class

@dataclass # make changes in the config file
class TrainConfig(BaseConfig):


#the following are all in the templates file 
def autoenc_base(): # this also needs changing probably
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf


def cxr128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'cxrlmdb256' # maybe data needs to be changed here needs to be provided 
    conf.scale_up_gpus(4) #if 4 gpus are available
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def cxr128_autoenc_130M(): # This can probably stay the same 
    conf = cxr128_autoenc_base
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_autoenc_130M'
    return conf

def pretrain_cxr128_autoenc130M():
    conf = cxr128_autoenc_base()
    conf.pretrain = PretrainConfig(
        name='130M',
        path=f'checkpoints/{cxr128_autoenc_130M().name}/last.ckpt',
    )
    conf.latent_infer_path = f'checkpoints/{cxr128_autoenc_130M().name}/latent.pkl'
    return conf



# in templates latent

def cxr128_autoenc_latent():
    conf = pretrain_cxr128_autoenc130M()
    conf = latent_diffusion128_config(conf)
    conf = latent_mlp_2048_norm_10layers(conf)
    conf = latent_256_batch_size(conf)
    conf = adamw_weight_decay(conf)
    conf.total_samples = 101_000_000
    conf.latent_loss_type = LossType.l1
    conf.latent_beta_scheduler = 'const0.008'
    conf.name = 'ffhq128_autoenc_latent'
    return conf

#### Train DDIM ######################

# after changing trainconfig just data name sneed to be changed in the files 

############ TRAIN Classifier ###########################

#



