import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from tabulate import tabulate
from .base_dataset import build_dataset
from torch.utils.data import Dataset as TorchDataset
import logging
logger: logging.Logger

def build_data_loader(
    cfg,
    data_source=None,
    batch_size=64,
    shuffle=True,
    is_train=True,
    dataset_wrapper=None
):
    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, is_train=is_train),
        batch_size=batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=(torch.cuda.is_available() and cfg.ENV.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        assistant,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(assistant)
        cfg = assistant.cfg

        # Build train_loader
        train_loader = build_data_loader(
            cfg,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            shuffle=True,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.show_dataset_summary(assistant)

    def show_dataset_summary(self, assistant):
        dataset_name = assistant.cfg.DATASET.NAME

        table = []
        table.append(["Dataset", dataset_name])
        table.append(["# train", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        logger.info(f"Dataset summary:\n{tabulate(table)}")


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.is_train = is_train

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {"index": idx}
        output.update(item.to_dict())
        
        return output