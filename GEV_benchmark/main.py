import os
from opt import *
import torch
import random
import numpy as np
from runner import Runner


import yaml
import time 
import utils
import sys

def main():
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # 保证可以复现的设定
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)  
    
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False
    
    # set up experiment 
    
    if args.exp_name == 'debug' or args.exp_name == 'test':
        args.experiment_path = os.path.join('./exp',args.exp_name)
    else:
        current_time = time.localtime(time.time())
        yy, mm, dd, hh, mi = current_time.tm_year, current_time.tm_mon, current_time.tm_mday, current_time.tm_hour, current_time.tm_min
        args.experiment_path = os.path.join('./exp','{}_{:02d}{:02d}{:02d}{:02d}{:02d}'.format(args.exp_name, int(yy), int(mm), int(dd), int(hh), int(mi)))
    
    if not os.path.isdir(args.experiment_path):
        os.makedirs(args.experiment_path)

    config_path = os.path.join(args.experiment_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)
        
    
    if args.running_mode == 'train':
        runner = Runner(args)
        utils.back_up(args.experiment_path)
        runner.train()
    elif args.running_mode == 'test':
        from test_runner import Test_Runner
        runner = Test_Runner(args)
        runner.test()
    return

if __name__ == '__main__':

    main()