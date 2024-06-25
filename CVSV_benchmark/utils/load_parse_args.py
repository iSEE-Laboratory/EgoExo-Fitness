import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, default='resnet', help='choose one backbone (s3dg or resnet)')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size of per gpu, default is 4')
    parser.add_argument('--warmup_step', type=int, default=0, help='set the warmup step size, default is 0')
    parser.add_argument('--lr', type=int, default=0.0005, help='learning rate, default is 1e-4')
    parser.add_argument('--num_sample', type=int, default=1600, help='pairs num sampled from pair txt, default is 1600')
    parser.add_argument('--num_clip', type=int, default=16, help='frames num sampled from one video, default is 16')
    parser.add_argument('--num_workers', type=int, default=10, help='num_worker, default is 10')
    parser.add_argument('--save_epochs', type=int, default=100, help='save epochs frequency, default is 100')
    parser.add_argument('--max_epoch', type=int, default=300, help='the max epochs, default is 300')
    parser.add_argument('--seq_loss', type=int, default=1.0, help='the W of seq loss, default: 1')
    parser.add_argument('--info_loss', type=int, default=1.0, help='the W of seq loss, default: 1')
    parser.add_argument('--dist', type=str, default='NormL2', help='the distance of inference final scores ')

    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--unfreeze_backbone', dest='freeze_backbone', action='store_false')
    parser.set_defaults(freeze_backbone=True)

    parser.add_argument('--freeze_BN', action='store_true')
    parser.add_argument('--unfreeze_BN', dest='freeze_BN', action='store_false')
    parser.set_defaults(freeze_BN=False)

    parser.add_argument('--random_sample', action='store_true')
    parser.add_argument('--uniform_sample', dest='random_sample', action='store_false')
    parser.set_defaults(random_sample=True)

    parser.add_argument('--pre_load', action='store_true')
    parser.add_argument('--unpre_load', dest='pre_load', action='store_false')
    parser.set_defaults(pre_load=False)

    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--unpair', dest='pair', action='store_false')
    parser.set_defaults(pair=True)

    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--undata_aug', dest='data_aug', action='store_false')
    parser.set_defaults(data_aug=False)

    parser.add_argument('--multi_TE', action='store_true')
    parser.add_argument('--single_TE', dest='multi_TE', action='store_false')
    parser.set_defaults(multi_TE=False)

    # --------------------------------------------------------------
    # gumble softmax
    parser.add_argument('--use_gumbel', action='store_true')
    parser.add_argument('--unuse_gumbel', dest='use_gumbel',  action='store_false')
    parser.set_defaults(use_gumbel=True)

    parser.add_argument('--gt_type', type=str, default='sort', help='the type of gumbel loss gt type')
    # --------------------------------------------------------------
    parser.add_argument('--warmup_LR', action='store_true')
    parser.add_argument('--unwarmup_LR', dest='warmup_LR', action='store_false')
    parser.set_defaults(warmup_LR=False)

    # retrival
    parser.add_argument('--retrival', action='store_true')
    parser.add_argument('--unretrival', dest='retrival', action='store_false')
    parser.set_defaults(retrival=False)

    # sim matrix
    parser.add_argument('--use_sim', action='store_true')
    parser.add_argument('--unuse_sim', dest='use_sim', action='store_false')
    parser.set_defaults(use_sim=False)

    parser.add_argument('--sim_dist',  default='l2', help='choose one sim distance (l2 or attn)')

    parser.add_argument('--concat_sim', action='store_true')
    parser.add_argument('--unconcat_sim', dest='concat_sim', action='store_false')
    parser.set_defaults(concat_sim=False)

    parser.add_argument('--add_sim', action='store_true')
    parser.add_argument('--unadd_sim', dest='add_sim', action='store_false')
    parser.set_defaults(add_sim=False)

    parser.add_argument('--sim_pos', default='first', help='set the pos of sim matrix')
    # --------------------------------------------------------------
    # language loss
    parser.add_argument('--use_text', action='store_true')
    parser.add_argument('--unuse_text', dest='use_text', action='store_false')
    parser.set_defaults(use_text=False)

    # pool_sent
    parser.add_argument('--pool_sent', action='store_true')
    parser.add_argument('--unpool_sent', dest='pool_sent', action='store_false')
    parser.set_defaults(pool_sent=False)


    parser.add_argument('--use_neg_text', action='store_true')
    parser.add_argument('--unuse_neg_text', dest='use_neg_text', action='store_false')
    parser.set_defaults(use_neg_text=False)

    parser.add_argument('--info_mask', action='store_true')
    parser.add_argument('--uninfo_mask', dest='info_mask', action='store_false')
    parser.set_defaults(info_mask=False)

    parser.add_argument('--info_ddp', action='store_true')
    parser.add_argument('--uninfo_ddp', dest='info_ddp', action='store_false')
    parser.set_defaults(info_ddp=False)

    # --------------------------------------------------------------

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--uneval', dest='eval', action='store_false')
    parser.set_defaults(eval=True)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--undebug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--unuse_ddp', dest='use_ddp', action='store_false')
    parser.set_defaults(use_ddp=True)
    parser.add_argument("--local_rank", default=-1, type=int)

    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--unuse_amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)


    parser.add_argument('--cfg_from_args', action='store_true')
    parser.add_argument('--uncfg_from_args', dest='cfg_from_args', action='store_false')
    parser.set_defaults(cfg_from_args=False)

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--unsave_model', dest='save_model', action='store_false')
    parser.set_defaults(save_model=False)

    parser.add_argument('--config', default='configs/train_config.yml', help='config file path')
    parser.add_argument('--eval_config', default='configs/eval_resnet_config.yml', help='config file path')
    parser.add_argument('--save_path', default=None, help='path to save models and log')
    parser.add_argument('--load_path', default=None, help='path to load the model')
    parser.add_argument('--log_name', default='train_log', help='log name')
    parser.add_argument('--tensorboard', required=False, default='default', help='tensorboard log name cannot be blank')

    args = parser.parse_args()
    return args

