# src/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from tqdm import tqdm
from typing import Dict

# from project files
from src.datasets.datasets import SegmentationDataset
from src.metrics.iou import compute_iou
from src.utils.utils import calculate_class_weights, visualize_predictions
from src.transforms.train_transform import train_transform
from src.transforms.val_test_transform import val_test_transform
from src.models.linknet import Linknet
# from src.models.unet import Unet
from src.loss.bce import BCEWithLogitsLoss

@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available else "cpu")


    # data

    # components

    # logger

    # functions
    # train
    # val

    # train