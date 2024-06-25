from yacs.config import CfgNode as CN

# 创建一个配置节点_C
_C = CN()


# default configuration

# Train configuration
_C.TRAIN = CN()
_C.TRAIN.SEED = 1234
_C.TRAIN.USE_CUDA = True
_C.TRAIN.MAX_EPOCH = 120
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.LR = 0.0001
_C.TRAIN.DROPOUT = 0.0
_C.TRAIN.SAVE_PATH = 'csv_logs'
_C.TRAIN.FREEZE_BACKBONE = False
_C.TRAIN.WEAK_SUPERVISION = False

# Model configuration
_C.MODEL = CN()
_C.MODEL.BACKBONE = 'cat'
_C.MODEL.BASE_MODEL = 'resnet50'
_C.MODEL.PRETRAIN = '/home/featurize/work/yuanming/pretrained_models/resnet50-kinetics400.pth'        # pretrain model path
_C.MODEL.DIM_EMBEDDING = 128
_C.MODEL.TRANSFORMER = False    # whether to use ViT module
_C.MODEL.ALIGNMENT = False      # whether to use sequence alignment module
_C.MODEL.SEQ_LOSS_COEF = 1.0
_C.MODEL.INFO_LOSS_COEF = 1.0
_C.MODEL.GUMBEL_LOSS_COEF = 1.0
_C.MODEL.SAVE_EPOCHS = 1        # save model per 1 epochs
_C.MODEL.SAVE_MODEL_LOG = True        # save model per 1 epochs


# Dataset configuration
_C.DATASET = CN()
_C.DATASET.NAME = 'COIN'
_C.DATASET.TXT_PATH = '/p300/dataset/ActionVerification/train.txt'
_C.DATASET.NUM_WORKERS = 0
_C.DATASET.MODALITY = 'RGB'
_C.DATASET.NUM_CLASS = 20       # 20 classes in total
_C.DATASET.MODE = 'train'       # train, test, val
_C.DATASET.NUM_SAMPLE = 600     # num of training samples for each epoch
_C.DATASET.AUGMENT = False       # whether to apply Data augmentation
_C.DATASET.NUM_CLIP = 8         # num of clips
_C.DATASET.SHUFFLE = True       # whether to shuffle the Data
_C.DATASET.RANDOM_SAMPLE = True       # whether to shuffle the Data



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # 克隆一份配置节点_C的信息返回，_C的信息不会改变
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

