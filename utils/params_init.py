import argparse

class SyncAction(argparse.Action):
    def __init__(self, sync_to, **kwargs):
        """
        :param sync_to: 目标参数名（如将当前参数值同步到args.d_model）
        :param kwargs: 传递给父类的其他参数（如dest、help等）
        """
        self.sync_to = sync_to
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # 1. 将当前参数值存储到原始dest（如args.d_model_c）
        setattr(namespace, self.dest, values)
        # 2. 同步值到目标参数（如args.d_model）
        setattr(namespace, self.sync_to, values)

def get_args():
    """
    获取命令行参数
    :return: args
    """

    parser = argparse.ArgumentParser(
        description='D3U-Framework-for-Probabilistic-MTS-Forecastingg',
        )

    # basic config
    parser.add_argument('--is_training', action='store_true', help='status')
    parser.add_argument('--model_id', type=str, default='ETTh2_96_192', help='model id')
    parser.add_argument('--model', type=str, default='PatchTST',
                        help='model name, options: [ns_Transformer, SVQ]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--data_name', type=str, default='custom', help='ETTh2')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    ##Model save path: checkpoints, pre-trained conditional network model save path: pretrain_checkpoints
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pretrain_checkpoints', type=str, default='./pretrain_checkpoints/', help='location of condition model checkpoints')
    # CART related args
    parser.add_argument('--diffusion_config_dir', type=str, default='./model9_NS_transformer/configs/toy_8gauss.yml',
                        help='')
    
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    ##Condition Network Model Parameter
    parser.add_argument('--d_model_c', action=SyncAction, sync_to="d_model", type=int, default=512, 
                        help='dimension of condition_model')
    parser.add_argument('--n_heads_c', action=SyncAction, sync_to="n_heads", type=int, default=8, 
                        help='num of heads')
    parser.add_argument('--e_layers_c', action=SyncAction, sync_to="e_layers", type=int, default=2, 
                        help='num of encoder layers')
    parser.add_argument('--d_layers_c', action=SyncAction, sync_to="d_layers", type=int, default=1, 
                        help='num of decoder layers')
    parser.add_argument('--factor_c', action=SyncAction, sync_to="factor", type=int, default=3, 
                        help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', 
                        help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', 
                        help='whether to predict unseen future data')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    # Series Decomposition (Optional, for case study1)
    parser.add_argument('--decomposition', action='store_true', help='decomposition')
    parser.add_argument('--kernel_size', type=int, default=15, help='kernel_size length of moving average')
    parser.add_argument('--fourier_factor', type=float, default=1.0, help='factor in computing `top_k`')
    # Point Prediction Model params
    parser.add_argument("--d_model", type=int, default=512, help="dimension of condition_model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads in condition_model")
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    # NSformer
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # Sparse-VQ
    parser.add_argument('--wFFN', type=int, default=1, help='use FFN layer')
    parser.add_argument('--svq', type=int, default=1, help='use sparse vector quantized')
    parser.add_argument('--codebook_size', type=int, default=128, help='codebook_size in sparse vector quantized')
    parser.add_argument('--sout', type=int, default=0, help='sparse linear for output')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--length', type=int, default=96)
    parser.add_argument('--num_codebook', type=int, default=4, 
                        help='number of codebooks in sparse vector quantized')
    # iTransformer
    parser.add_argument('--use_norm', type=int, default=1, help='use norm and denorm')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')

    ##Denoising Network Model Parameter
    parser.add_argument('--denoise_model', type=str, default='PatchDN', help='denoise_model')
    # PatchDN
    parser.add_argument('--patch_size', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--depth', type=int, default=1, help='depth')
    parser.add_argument('--d_model_d', type=int, default=512, help='dimension of denoise model')
    parser.add_argument('--n_heads_d', type=int, default=8, help='num of heads')
    # U-Net
    parser.add_argument('--ddpm_inp_embed', type=int, default=256)
    parser.add_argument('--ddpm_dim_diff_steps', type=int, default=256)
    parser.add_argument('--ddpm_channels_conv', type=int, default=256)
    parser.add_argument('--ddpm_channels_fusion_I', type=int, default=256)
    parser.add_argument('--ddpm_layers_inp', type=int, default=5)
    parser.add_argument('--ddpm_layers_I', type=int, default=5)
    parser.add_argument('--ddpm_layers_II', type=int, default=5)
    parser.add_argument('--cond_ddpm_num_layers', type=int, default=5)
    parser.add_argument('--cond_ddpm_channels_conv', type=int, default=64)

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  # 32
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of train input data')  # 32
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate for diffusion model')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Some args for Ax (all about diffusion part)
    parser.add_argument('--timesteps', type=int, default=1000, help='')
    parser.add_argument('--sampling_timesteps', type=int, default=50, help='')
    parser.add_argument('--fastsample', action='store_true', help='')
    parser.add_argument('--type_sampler', type=str, default='None', help='loss function')  
    parser.add_argument('--DPMsolver_step', type=int, default=20, help='')
    parser.add_argument('--eta', type=float, default=1, help='')
    parser.add_argument('--parameterization', type=str, default="noise", help='')
    parser.add_argument('--bias', action='store_true', help='')
    parser.add_argument('--use_pretraining_condition', action='store_true', help='')

    # rebuttal exps
    parser.add_argument('--cond_pred_model_requires_grad', action='store_true', help='') #对应实验微调
    parser.add_argument('--from_scrach', action='store_true', help='')#对应实验 从0开始训练
    parser.add_argument('--scrach_10_stop', action='store_true', help='')#对应实验 从0开始训练
    parser.add_argument('--bias_y_0', action='store_true', help='') #对应实验 条件模型建模T+S,扩散模型建模残差

    args = parser.parse_args()

    return args
