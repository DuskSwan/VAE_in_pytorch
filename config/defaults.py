from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DEVICE = "cuda"
_C.SEED = 42

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.IMG_LENGTH = 128
_C.MODEL.HIDDEN_CHNLS = [16, 32, 64, 128, 256]
_C.MODEL.LATENT_DIM = 128

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.CHECKPOINT_PERIOD = 10
_C.TRAIN.NEED_CHRCKPOINT = False
_C.TRAIN.DATA_PATH = r'data\datasets\CelebA'

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.BATCH_SIZE = 4
_C.INFERENCE.MODEL_PATH = r'output\model.pth'
_C.INFERENCE.DATA_PATH = r'data\datasets\CelebA'

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.KL_WEIGHT = 0.00025

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# -----------------------------------------------------------------------------
# LOG
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.DIR = "./log"
_C.LOG.ITER_INTERVAL = 1
_C.LOG.EPOCH_INTERVAL = 10
_C.LOG.OUTPUT_TO_FILE = True # 是否输出到文件，默认输出到控制台
_C.LOG.PREFIX = "default" # 输出到文件的命名前缀
