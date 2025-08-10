from yacs.config import CfgNode as CN
from base import BaseConfig


class CodeTalkerConfig(BaseConfig):
  def __init__(self, cfg_path=None, new_allowed=True):
    super().__init__(new_allowed)

    # Add specific configurations for CodeTalker here
    self.cfg.DATASET.READ_AUDIO = True
    self.cfg.DATASET.TRAIN_SUBJECTS = []
    self.cfg.DATASET.VAL_SUBJECTS = []
    self.cfg.DATASET.TEST_SUBJECTS = []
   
    self.cfg.DATASET.SPLIT = CN()
    self.cfg.DATASET.SPLIT.VOCASET = CN()
    self.cfg.DATASET.SPLIT.VOCASET.TRAIN = list(range(1, 41))
    self.cfg.DATASET.SPLIT.VOCASET.VAL = list(range(21, 41))
    self.cfg.DATASET.SPLIT.VOCASET.TEST = list(range(21, 41))

    self.cfg.DATASET.SPLIT.BIWI = CN()
    self.cfg.DATASET.SPLIT.BIWI.TRAIN = list(range(1, 33))
    self.cfg.DATASET.SPLIT.BIWI.VAL = list(range(33, 37))
    self.cfg.DATASET.SPLIT.BIWI.TEST = list(range(37, 41))

    self.cfg.OPTIM.PATIENCE = 3
    self.cfg.OPTIM.THRESHOLD = 0.0001
    self.cfg.OPTIM.POLY_LR = False
    self.cfg.OPTIM.POWER = 0.9

    self.cfg.TRAIN.PRINT_FREQ = 5


    
    if cfg_path is not None:
      self.cfg.merge_from_file(cfg_path)

    self.system_init()
