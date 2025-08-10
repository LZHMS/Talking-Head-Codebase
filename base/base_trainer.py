
import shutil
import pickle
import os.path as osp
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import (
  tolist_if_not, mkdir_if_missing, Registry, check_availability,
  RAdam, ConstantWarmupScheduler, LinearWarmupScheduler, calc_vq_loss, calc_logit_loss)

TRAINER_REGISTRY = Registry("TRAINER")

def build_trainer(assistant):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(assistant.cfg.TRAINER.NAME, avai_trainers)
    if assistant.cfg.ENV.VERBOSE:
        assistant.logger.info("Loading trainer: {}".format(assistant.cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(assistant.cfg.TRAINER.NAME)(assistant)

class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self, assistant):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

        self.check_cfg(assistant.cfg)
        self.device = assistant.device
        self.logger = assistant.logger

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    """Writer for TensorBoard.
        Functions:
            > init_writer
            > close_writer
            > write_scalar
    """
    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            self.logger.info(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    """Train model with a generic training loop.
        Functions:
            > train
            > before_train
            > after_train
            > before_epoch
            > after_epoch
            > run_epoch
            > parse_batch_train
    """
    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    """Test model with a generic testing loop.
        Functions:
            > test
            > parse_batch_test
    """
    def test(self):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    """Model update with a generic forward-backward loop.
        Functions:
            > build_loss_metrics
            > forward_backward
            > model_inference
            > model_zero_grad
            > model_backward
            > model_update
            > update_lr
            > model_backward_and_update
            > detect_anomaly
    """
    def build_loss_metrics(self, loss_fc_name):
        if loss_fc_name == "VQLoss":
            self.logger.info("Using VQ loss function for metrics ...")
            return calc_vq_loss

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def build_optimizer(self, model, param_groups=None):
        """A function wrapper for building an optimizer.

        Args:
            model (nn.Module or iterable): model.
            optim_cfg (CfgNode): optimization config.
            param_groups: If provided, directly optimize param_groups and abandon model
        """
        AVAI_OPTIMS = ["adam", "amsgrad", "sgd", "rmsprop", "radam", "adamw"]
        optim = self.assistant.cfg.OPTIM.NAME
        lr = self.assistant.cfg.OPTIM.LR
        weight_decay = self.assistant.cfg.OPTIM.WEIGHT_DECAY
        momentum = self.assistant.cfg.OPTIM.MOMENTUM
        sgd_dampening = self.assistant.cfg.OPTIM.SGD_DAMPNING
        sgd_nesterov = self.assistant.cfg.OPTIM.SGD_NESTEROV
        rmsprop_alpha = self.assistant.cfg.OPTIM.RMSPROP_ALPHA
        adam_beta1 = self.assistant.cfg.OPTIM.ADAM_BETA1
        adam_beta2 = self.assistant.cfg.OPTIM.ADAM_BETA2
        staged_lr = self.assistant.cfg.OPTIM.STAGED_LR
        new_layers = self.assistant.cfg.OPTIM.NEW_LAYERS
        base_lr_mult = self.assistant.cfg.OPTIM.LR_SCHEDULER

        if optim not in AVAI_OPTIMS:
            raise ValueError(
                f"optim must be one of {AVAI_OPTIMS}, but got {optim}"
            )

        if param_groups is not None and staged_lr:
            self.logger.warning(
                "staged_lr will be ignored, if you need to use staged_lr, "
                "please bind it with param_groups yourself."
            )

        if param_groups is None:
            if staged_lr:
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        "When staged_lr is True, model given to "
                        "build_optimizer() must be an instance of nn.Module"
                    )

                if isinstance(model, nn.DataParallel):
                    model = model.module

                if isinstance(new_layers, str):
                    if new_layers is None:
                        self.logger.warning("new_layers is empty (staged_lr is useless)")
                    new_layers = [new_layers]

                base_params = []
                base_layers = []
                new_params = []

                for name, module in model.named_children():
                    if name in new_layers:
                        new_params += [p for p in module.parameters()]
                    else:
                        base_params += [p for p in module.parameters()]
                        base_layers.append(name)

                param_groups = [
                    {
                        "params": base_params,
                        "lr": lr * base_lr_mult
                    },
                    {
                        "params": new_params
                    },
                ]

            else:
                if isinstance(model, nn.Module):
                    param_groups = model.parameters()
                else:
                    param_groups = model

        if optim == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )

        elif optim == "amsgrad":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
                amsgrad=True,
            )

        elif optim == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=sgd_dampening,
                nesterov=sgd_nesterov,
            )

        elif optim == "rmsprop":
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                alpha=rmsprop_alpha,
            )

        elif optim == "radam":
            optimizer = RAdam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )

        elif optim == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_beta1, adam_beta2),
            )
        else:
            raise NotImplementedError(f"Optimizer {optim} not implemented yet!")

        return optimizer
    
    def build_lr_scheduler(self, optimizer):
        """A function wrapper for building a learning rate scheduler.

        Args:
            optimizer (Optimizer): an Optimizer.
            optim_cfg (CfgNode): optimization config.
        """
        AVAI_SCHEDS = ["single_step", "multi_step", "cosine"]

        lr_scheduler = self.assistant.cfg.OPTIM.LR_SCHEDULER
        step_size = self.assistant.cfg.OPTIM.STEP_SIZE
        gamma = self.assistant.cfg.OPTIM.GAMMA
        max_epoch = self.assistant.cfg.OPTIM.MAX_EPOCH
        warmup_epoch = self.assistant.cfg.OPTIM.WARMUP_EPOCH
        warmup_type = self.assistant.cfg.OPTIM.WARMUP_TYPE
        warmup_recount = self.assistant.cfg.OPTIM.WARMUP_RECOUNT
        warmup_cons_lr = self.assistant.cfg.OPTIM.WARMUP_CONS_LR
        warmup_min_lr = self.assistant.cfg.OPTIM.WARMUP_MIN_LR

        if lr_scheduler not in AVAI_SCHEDS:
            raise ValueError(
                f"scheduler must be one of {AVAI_SCHEDS}, but got {lr_scheduler}"
            )

        if lr_scheduler == "single_step":
            if isinstance(step_size, (list, tuple)):
                step_size = step_size[-1]

            if not isinstance(step_size, int):
                raise TypeError(
                    "For single_step lr_scheduler, step_size must "
                    f"be an integer, but got {type(step_size)}"
                )

            if step_size <= 0:
                step_size = max_epoch

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

        elif lr_scheduler == "multi_step":
            if not isinstance(step_size, (list, tuple)):
                raise TypeError(
                    "For multi_step lr_scheduler, step_size must "
                    f"be a list, but got {type(step_size)}"
                )

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size, gamma=gamma
            )

        elif lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(max_epoch)
            )

        if warmup_epoch > 0:
            if not warmup_recount:
                scheduler.last_epoch = warmup_epoch

            if warmup_type == "constant":
                scheduler = ConstantWarmupScheduler(
                    optimizer, scheduler, warmup_epoch,
                    warmup_cons_lr
                )

            elif warmup_type == "linear":
                scheduler = LinearWarmupScheduler(
                    optimizer, scheduler, warmup_epoch,
                    warmup_min_lr
                )

            else:
                raise ValueError

        return scheduler
    """Save the model at a given directory.
        Functions:
            > save_model
            > save_checkpoint
    """
    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            self.save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def save_checkpoint(
        self,
        state,
        save_dir,
        is_best=False,
        remove_module_from_keys=True,
        model_name=""
    ):
        r"""Save checkpoint.

        Args:
            state (dict): dictionary.
            save_dir (str): directory to save checkpoint.
            is_best (bool, optional): if True, this checkpoint will be copied and named
                ``model-best.pth.tar``. Default is False.
            remove_module_from_keys (bool, optional): whether to remove "module."
                from layer names. Default is True.
            model_name (str, optional): model name to save.
        """
        mkdir_if_missing(save_dir)

        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            state["state_dict"] = new_state_dict

        # save model
        epoch = state["epoch"]
        if not model_name:
            model_name = "model.pth.tar-" + str(epoch)
        fpath = osp.join(save_dir, model_name)
        torch.save(state, fpath)
        self.logger.info(f"Checkpoint saved to {fpath}")

        # save current model name
        checkpoint_file = osp.join(save_dir, "checkpoint")
        checkpoint = open(checkpoint_file, "w+")
        checkpoint.write("{}\n".format(osp.basename(fpath)))
        checkpoint.close()

        if is_best:
            best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
            shutil.copy(fpath, best_fpath)
            self.assistant.logger.info('Best checkpoint saved to "{}"'.format(best_fpath))

    """Load a checkpoint from a given directory.
        Functions:
            > load_model
            > load_checkpoint
            > load_pretrained_weights
    """
    def load_model(self, directory, epoch=None):
        if not directory:
            self.logger.warning(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = self.load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            self.logger.info(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)
    
    def load_checkpoint(self, fpath):
        r"""Load checkpoint.

        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.

        Args:
            fpath (str): path to checkpoint.

        Returns:
            dict

        Examples::
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)
        """
        if fpath is None:
            raise ValueError("File path is None")

        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = "cpu" if self.device == "cpu" else None

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            self.logger.error('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint

    def load_pretrained_weights(self, model, weight_path):
        r"""Load pretrianed weights to model.

        Features::
            - Incompatible layers (unmatched in name or size) will be ignored.
            - Can automatically deal with keys containing "module.".

        Args:
            model (nn.Module): network model.
            weight_path (str): path to pretrained weights.

        Examples::
            >>> weight_path = 'log/my_model/model-best.pth.tar'
            >>> load_pretrained_weights(model, weight_path)
        """
        checkpoint = self.load_checkpoint(weight_path)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # discard module.

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            self.logger.warning(
                f"Cannot load {weight_path} (check the key names manually)"
            )
        else:
            self.logger.info(f"Successfully loaded pretrained weights from {weight_path}")
            if len(discarded_layers) > 0:
                self.logger.info(
                    f"Layers discarded due to unmatched keys or size: {discarded_layers}"
                )

    """Resume model training from a checkpoint if it exists.
        Functions:
            > resume_model_if_exist
            > resume_from_checkpoint
    """
    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            self.logger.warning("No checkpoint found, train from scratch")
            return 0

        self.logger.info(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = self.resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def resume_from_checkpoint(self, fdir, model, optimizer=None, scheduler=None):
        r"""Resume training from a checkpoint.

        This will load (1) model weights and (2) ``state_dict``
        of optimizer if ``optimizer`` is not None.

        Args:
            fdir (str): directory where the model was saved.
            model (nn.Module): model.
            optimizer (Optimizer, optional): an Optimizer.
            scheduler (Scheduler, optional): an Scheduler.

        Returns:
            int: start_epoch.

        Examples::
            >>> fdir = 'log/my_model'
            >>> start_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
        """
        with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
            model_name = checkpoint.readlines()[0].strip("\n")
            fpath = osp.join(fdir, model_name)

        self.assistant.logger.info('Loading checkpoint from "{}"'.format(fpath))
        checkpoint = self.load_checkpoint(fpath)
        model.load_state_dict(checkpoint["state_dict"])
        self.assistant.logger.info("Loaded model weights")

        if optimizer is not None and "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.assistant.logger.info("Loaded optimizer")

        if scheduler is not None and "scheduler" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler"])
            self.assistant.logger.info("Loaded scheduler")

        start_epoch = checkpoint["epoch"]
        self.assistant.logger.info("Previous epoch: {}".format(start_epoch))

        return start_epoch
    
    """Some tools for parameters calculation.
        Functions:
            > count_num_param
    """
    def count_num_param(self, model=None, params=None):
        r"""Count number of parameters in a model.

        Args:
            model (nn.Module): network model.
            params: network model`s params.
        Examples::
            >>> model_size = count_num_param(model)
        """

        if model is not None:
            return sum(p.numel() for p in model.parameters())

        if params is not None:
            s = 0
            for p in params:
                if isinstance(p, dict):
                    s += p["params"].numel()
                else:
                    s += p.numel()
            return s

        raise ValueError("model and params must provide at least one.")