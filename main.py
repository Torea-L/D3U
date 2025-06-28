import random

import numpy as np
import setproctitle
import torch
from models.exp.exp_main import Exp_Main
from utils import params_init

if __name__ == '__main__':
    setproctitle.setproctitle('D3U_thread')

    args = params_init.get_args()

    if args.seed == -1:
        fix_seed = np.random.randint(2147483647)
    else:
        fix_seed = args.seed

    print('Using seed:', fix_seed)

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        else:
            torch.cuda.set_device(args.gpu)
    print(f'Args in experiment :\n {args}')

    Exp = Exp_Main
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_ts{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.decomposition,
                args.timesteps,
                args.denoise_model,
                args.model_id,
                args.model,
                args.data_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model_c,
                args.n_heads_c,
                args.e_layers_c,
                args.d_layers_c,
                args.d_ff,
                args.factor_c,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test_cond(setting)
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_ts{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.decomposition,
            args.timesteps,
            args.denoise_model,
            args.model_id,
            args.model,
            args.data_name,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model_c,
            args.n_heads_c,
            args.e_layers_c,
            args.d_layers_c,
            args.d_ff,
            args.factor_c,
            args.embed,
            args.distil,
            args.des, ii 
        )
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


