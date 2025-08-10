import logging
import random
import numpy as np
import torch
from yacs.config import CfgNode as CN


class BaseConfig:
  def __init__(self, new_allowed=True):
    ###########################
    # Config definition
    ###########################
    cfg = CN(new_allowed=new_allowed)

    ###########################
    # Env
    ###########################
    cfg.ENV = CN(new_allowed=new_allowed)
    cfg.ENV.VERSION = 1
    cfg.ENV.SEED = -1
    # Directory to save the output files (like log.txt and model weights)
    cfg.ENV.OUTPUT_DIR = "./output"
    # Path to a directory where the files were saved previously
    cfg.ENV.RESUME = ""
    cfg.ENV.GPU = [0]
    cfg.ENV.USE_CUDA = True
    # Print detailed information
    # E.g. trainer, dataset, and backbone
    cfg.ENV.VERBOSE = True
    # Name and description of the experiment 
    cfg.ENV.NAME = ""
    cfg.ENV.DESCRIPTION = ""

    ###########################
    # Input
    ###########################
    cfg.INPUT = CN()
    # If True, tfm_train and tfm_test will be None
    cfg.INPUT.NO_TRANSFORM = False
    # Gaussian noise
    cfg.INPUT.GN_MEAN = 0.0
    cfg.INPUT.GN_STD = 0.15
    # RandomAugment
    cfg.INPUT.RANDAUGMENT_N = 2
    cfg.INPUT.RANDAUGMENT_M = 10

    ###########################
    # Dataset
    ###########################
    cfg.DATASET = CN()
    # Directory where datasets are stored
    cfg.DATASET.NAME = ""
    cfg.DATASET.ROOT = ""
    cfg.DATASET.AUDIO_PATH = ""
    cfg.DATASET.VERTICES_PATH = ""
    cfg.DATASET.TEMPLATE_FILE = ""
    # Percentage of validation data (only used for SSL datasets)
    # Set to 0 if do not want to use val data
    # Using val data for hyperparameter tuning was done in Oliver et al. 2018
    cfg.DATASET.VAL_PERCENT = 0.1

    ###########################
    # Dataloader
    ###########################
    cfg.DATALOADER = CN()
    cfg.DATALOADER.NUM_WORKERS = 4
    # Setting for the train data-loader
    cfg.DATALOADER.TRAIN = CN()
    cfg.DATALOADER.TRAIN.BATCH_SIZE = 32

    # Setting for the test data-loader
    cfg.DATALOADER.TEST = CN()
    cfg.DATALOADER.TEST.BATCH_SIZE = 32

    ###########################
    # Model
    ###########################
    cfg.MODEL = CN()
    cfg.MODEL.NAME = ""
    # Path to model weights (for initialization)
    cfg.MODEL.INIT_WEIGHTS = ""
    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = ""
    cfg.MODEL.BACKBONE.PRETRAINED = True
    cfg.MODEL.BACKBONE.IN_DIM = 15069
    cfg.MODEL.BACKBONE.HIDDEN_SIZE = 1024
    cfg.MODEL.BACKBONE.NUM_HIDDEN_LAYERS = 6
    cfg.MODEL.BACKBONE.NUM_ATTENTION_HEADS = 8
    cfg.MODEL.BACKBONE.INTERMEDIATE_SIZE = 1536
    cfg.MODEL.BACKBONE.WINDOW_SIZE = 1
    # for VQ-VAE config
    cfg.MODEL.BACKBONE.QUANT_FACTOR = 0
    cfg.MODEL.BACKBONE.FACE_QUAN_NUM = 16
    cfg.MODEL.BACKBONE.NEG = 0.2
    cfg.MODEL.BACKBONE.INAFFINE = False
    # Definition of embedding layers
    cfg.MODEL.HEAD = CN()
    # If none, do not construct embedding layers, the
    # backbone's output will be passed to the classifier
    cfg.MODEL.HEAD.NAME = ""
    # Structure of hidden layers (a list), e.g. [512, 512]
    # If undefined, no embedding layer will be constructed
    cfg.MODEL.HEAD.HIDDEN_LAYERS = ()
    cfg.MODEL.HEAD.ACTIVATION = "relu"
    cfg.MODEL.HEAD.BN = True
    cfg.MODEL.HEAD.DROPOUT = 0.0
    # VQ-VAE config
    cfg.MODEL.HEAD.N_EMBED = 256
    cfg.MODEL.HEAD.ZQUANT_DIM = 64
    cfg.MODEL.WAV2VEC2_PATH = ""

    ###########################
    # Optimization
    ###########################
    cfg.OPTIM = CN()
    cfg.OPTIM.NAME = "adam"
    cfg.OPTIM.LR = 0.0003
    cfg.OPTIM.WEIGHT_DECAY = 5e-4
    cfg.OPTIM.MOMENTUM = 0.9

    cfg.OPTIM.STEP_LR = True
    cfg.OPTIM.ADAPTIVE_LR = False
    cfg.OPTIM.FACTOR = 0.3
    

    # sgd
    cfg.OPTIM.SGD_DAMPNING = 0
    cfg.OPTIM.SGD_NESTEROV = True

    cfg.OPTIM.RMSPROP_ALPHA = 0.99
    # The following also apply to other
    # adaptive optimizers like adamw
    cfg.OPTIM.ADAM_BETA1 = 0.9
    cfg.OPTIM.ADAM_BETA2 = 0.999
    # STAGED_LR allows different layers to have
    # different lr, e.g. pre-trained base layers
    # can be assigned a smaller lr than the new
    # classification layer
    cfg.OPTIM.STAGED_LR = False
    cfg.OPTIM.NEW_LAYERS = ()
    cfg.OPTIM.BASE_LR_MULT = 0.1


    # Learning rate scheduler
    cfg.OPTIM.LR_SCHEDULER = "single_step"
    cfg.OPTIM.STEP_SIZE = 20
    cfg.OPTIM.GAMMA = 0.5
    
    cfg.OPTIM.START_EPOCH = 0
    cfg.OPTIM.MAX_EPOCH = 100
    # Set WARMUP_EPOCH larger than 0 to activate warmup training
    cfg.OPTIM.WARMUP_EPOCH = -1
    cfg.OPTIM.WARMUP_STEPS = 1
    # Either linear or constant
    cfg.OPTIM.WARMUP_TYPE = "linear"
    # Constant learning rate when type=constant
    cfg.OPTIM.WARMUP_CONS_LR = 1e-5
    # Minimum learning rate when type=linear
    cfg.OPTIM.WARMUP_MIN_LR = 1e-5
    # Recount epoch for the next scheduler (last_epoch=-1)
    # Otherwise last_epoch=warmup_epoch
    cfg.OPTIM.WARMUP_RECOUNT = True

    ###########################
    # Loss
    ###########################
    cfg.LOSS = CN()
    cfg.LOSS.NAME = "VQLoss"
    cfg.LOSS.QUANT_LOSS_WEIGHT = 1.0

    ###########################
    # Trainer specifics
    ###########################
    cfg.TRAINER = CN()
    cfg.TRAINER.NAME = ""

    ###########################
    # Train
    ###########################
    cfg.TRAIN = CN()
    cfg.TRAIN.USE_SGD = False
    cfg.TRAIN.SYNC_BN = False  # adopt sync_bn or not
    # How often (epoch) to save model during training
    # Set to 0 or negative value to only save the last one
    cfg.TRAIN.SAVE_FREQ = 0
    # How often (batch) to print training information
    cfg.TRAIN.PRINT_FREQ = 10
    cfg.TRAIN.EVALUATE = True
    cfg.TRAIN.EVAL_FREQ = 10
    cfg.TRAIN.CHECKPOINT_FREQ = 1

    ###########################
    # Test
    ###########################
    cfg.TEST = CN()
    cfg.TEST.EVALUATOR = "Classification"
    # If NO_TEST=True, no testing will be conducted
    cfg.TEST.NO_TEST = False
    # Use test or val set for FINAL evaluation
    cfg.TEST.SPLIT = "test"
    # Which model to test after training (last_step or best_val)
    # If best_val, evaluation is done every epoch (if val data
    # is unavailable, test data will be used)
    cfg.TEST.FINAL_MODEL = "last_step"

    # OP
    self.cfg = cfg

  def system_init(self):
    # System Initialization
    ## logger configuration
    self.setup_logger()
    self.logger.info("Initializing main logger ...")

    ## random seed setting
    if self.cfg.ENV.SEED >= 0:
      self.logger.info('Setting fixed seed: {}'.format(self.cfg.ENV.SEED))
      self.set_random_seed(self.cfg.ENV.SEED)

    ## cuda setting
    if torch.cuda.is_available() and self.cfg.ENV.USE_CUDA:
      torch.backends.cudnn.benchmark = True
      self.device = torch.device(f"cuda:{self.cfg.ENV.GPU[0]}")
    else:
      self.device = torch.device("cpu")
      self.logger.info('Setting device to {}'.format(self.device))
  
  def setup_logger(self, logger_name="MainLogger"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = "[%(asctime)s %(filename)s line %(lineno)d]=>%(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    self.logger = logger

  def set_random_seed(self, seed):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

  def collect_env_info(self):
    """Return env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    return get_pretty_env_info()
  
  def print_info(self):
    """Print system info and env info.
    """
    self.logger.info('Collecting system info ...')
    self.logger.info(f"Project configuration:\n{self.cfg}")
    self.logger.info('Collecting env info ...')
    self.logger.info(f"Env information:\n{self.collect_env_info()}")