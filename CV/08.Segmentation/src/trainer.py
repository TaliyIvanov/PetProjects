from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import os
from tqdm import tqdm
from typing import Dict
from glob import glob
from pathlib import Path

# from project files
from src.datasets.datasets import SegmentationDataset
from src.metrics.iou import compute_iou
from src.utils.utils import calculate_class_weights, visualize_predictions
from src.transforms.train_transforms import train_transforms
from src.transforms.val_test_transforms import val_test_transforms
from src.models.linknet import Linknet
# from src.models.unet import Unet
from src.loss.bce import BCEWithLogitsLoss

def train(cfg: DictConfig) -> None:
    # 1.Base
    print("--- Конфигурация, полученная функцией train ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------------------------")

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available else "cpu")


    # 2.Data
    
    # data paths
    original_cwd = get_original_cwd()
    image_paths = sorted(glob(os.path.join(original_cwd, cfg.data.root_dir_images, "*.png")))
    masks_paths = sorted(glob(os.path.join(original_cwd, cfg.data.root_dir_masks, "*.png")))

    # train/val/test split
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(image_paths,
                                                                      masks_paths,
                                                                      test_size=cfg.data.val_size,
                                                                      random_state=cfg.data.random_state)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs,
                                                                  temp_masks,
                                                                  test_size=cfg.data.test_size,
                                                                  random_state=cfg.data.random_state)
    
    # datasets
    train_dataset = SegmentationDataset(train_imgs,
                                        train_masks,
                                        transform=hydra.utils.instantiate(cfg.transforms.train_transforms))
    val_dataset = SegmentationDataset(val_imgs,
                                      val_masks,
                                      transform=hydra.utils.instantiate(cfg.transforms.val_test_transforms))
    test_dataset = SegmentationDataset(test_imgs,
                                       test_masks,
                                       transform=hydra.utils.instantiate(cfg.transforms.val_test_transforms))

    # dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.data.batch_size,
                              shuffle=True,
                              num_workers=cfg.data.num_workers)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.data.batch_size,
                            shuffle=False,
                            num_workers=cfg.data.num_workers)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.data.batch_size,
                             shuffle=False,
                             num_workers=cfg.data.num_workers)

    # 3. Components
    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, params = model.parameters())
    scheduler = hydra.utils.instantiate(cfg.train.scheduler, optimizer=optimizer)
    loss_fn = hydra.utils.instantiate(cfg.train.loss_fn)
    # logger
    logger = hydra.utils.instantiate(cfg.logger)

    # 4. functions

    # train
    def train_epoch(model, dataloader, optimizer, loss_fn, device, logger, step):
        model.train()
        train_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[Epoch {step + 1}/{cfg.train.epochs}] Train")

        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.unsqueeze(1).float()

            # forward
            preds = model(images)
            loss = loss_fn(preds, masks)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_avg_train_loss = train_loss / (i + 1)
            pbar.set_postfix({"loss":running_avg_train_loss})

        avg_train_loss = train_loss / len(dataloader)
        logger.log_metrics({"train_loss":avg_train_loss}, step=step)
        return avg_train_loss

    # val
    def validate_epoch(model, dataloader, loss_fn, device, logger, step) -> Dict:
        model.eval()
        val_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks, in tqdm(dataloader, desc=f"[Epoch {step + 1}/{cfg.train.epochs}] Val"):
                images = images.to(device)
                masks = masks.to(device)
                masks = masks.unsqueeze(1).float()

                preds = model(images)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()

                # metrics
                preds_sigmoid = torch.sigmoid(preds)
                preds_binary = (preds_sigmoid > 0.5).float()
                iou = compute_iou(preds_binary, masks)
                total_iou  += iou

        avg_val_loss = val_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        logger.log_metrics({"val_loss":avg_val_loss, "val_iou": avg_iou}, step=step)
        return {"val_loss": avg_val_loss, "val_iou": avg_iou}

    def preds_on_test(model, test_loader, loss_fn, device):
        model.eval()
        test_iou = 0
        test_loss = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                masks = masks.unsqueeze(1).float()

                preds = model(images)
                loss = loss_fn(preds, masks)
                test_loss += loss.item()

                preds_sigmoid = torch.sigmoid(preds)
                preds_binary = (preds_sigmoid > 0.5).float()
                iou = compute_iou(preds_binary, masks)
                test_iou += iou.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_iou = test_iou / len(test_loader)

        return {"test_loss": avg_test_loss, "test_iou": avg_test_iou}

    
    # 5. Train loop
    model_filename = "best_model_linknet.pth"
    best_iou = 0.0
    for epoch in range(cfg.train.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, logger, epoch)
        metrics = validate_epoch(model, val_loader, loss_fn, device, logger, epoch)
        scheduler.step(metrics["val_loss"])
        if metrics["val_iou"] > best_iou:
            best_iou = metrics["val_iou"]

            torch.save(model.state_dict(), model_filename)
            print(f"model saved to {os.getcwd()}/{model_filename} (IoU improved to {best_iou:.4f})")
    
    best_model_state = torch.load(model_filename)
    model.load_state_dict(best_model_state)

    tests_results = preds_on_test(model, test_loader, loss_fn, device)
    print("Test results on the best model:")
    print(tests_results)

    logger.log_metrics(tests_results, step=cfg.train.epochs)
    logger.end()