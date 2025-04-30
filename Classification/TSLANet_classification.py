import argparse
import datetime
import os
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

# import webdataset as wds
# from pathlib import Path
from streaming import StreamingDataset

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timm.loss import LabelSmoothingCrossEntropy
from timm.layers import DropPath
from timm.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D
from network import model_pretraining, model_training


class LOBDataset(StreamingDataset):
    def __init__(self,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                ) -> None:
        super().__init__(
                         local=local, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         validate_hash=None)

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        x = obj['array'].copy()
        y = obj['class']
        return x, y


def pretrain_model(args, resume_checkpoint_path=None, loss_type="mse"):
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs

    trainer = L.Trainer(
        default_root_dir=PRETRAIN_CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("step"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(42)
    
    model = model_pretraining(loss_type, **args)

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint_path)

    return pretrain_checkpoint_callback.best_model_path

def train_model(args, pretrained_model_path=None, resume_checkpoint_path=None):
    trainer = L.Trainer(
        default_root_dir=TRAIN_CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=args.num_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("step"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(42)
    
    L.seed_everything(42)  # To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training(**args)

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint_path)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    conf_matrix = get_clf_report(model, test_loader, TRAIN_CHECKPOINT_PATH)

    return model, acc_result, f1_result, conf_matrix


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='AUDUSD')

    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_lr', type=float, default=1e-4)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--patch_size', type=int, default=8
    
    )

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    parser.add_argument("--resume_checkpoint", type=str2bool, default=False)
    args = parser.parse_args()

    # load from checkpoint
    if not args.resume_checkpoint:
        run_description = f"AUDUSD_CLF_dim{args.emb_dim}_depth{args.depth}_"
        run_description += f"ASB_{args.ASB}_AF_{args.adaptive_filter}_ICB_{args.ICB}_preTr_{args.load_from_pretrained}_patch_{args.patch_size}_batch_{args.batch_size}_"
        run_description += f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_description_train = run_description + "_train"
        run_description_pretrain = run_description + "_pre_train"

        print(f"========== {run_description} ===========")
        pretrain_model_path = None
        train_model_path = None
    else:
        run_description = None
        pretrain_model_path = None
        train_model_path = None

    PRETRAIN_CHECKPOINT_PATH = f"lightning_logs/{run_description_pretrain}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=PRETRAIN_CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    TRAIN_CHECKPOINT_PATH = f"lightning_logs/{run_description_train}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=TRAIN_CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(pretrain_checkpoint_callback)


    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    NUM_CLASSES = 9
    DATAPATH = f"/Users/danielfisher/repositories/futfut/mosaic_dataset_forward5000_backward5000_{NUM_CLASSES}bins"
    train_ds = LOBDataset(local=f"{DATAPATH}/train", batch_size=args.batch_size, shuffle=True)
    val_ds = LOBDataset(local=f"{DATAPATH}/val", batch_size=args.batch_size, shuffle=False)
    test_ds = LOBDataset(local=f"{DATAPATH}/test", batch_size=args.batch_size, shuffle=False)

    TRAIN_DS_SIZE = train_ds.size

    train_loader = DataLoader(train_ds,batch_size=args.batch_size, num_workers=1, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=args.batch_size, num_workers=1, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_ds,batch_size=args.batch_size, num_workers=1, persistent_workers=True, pin_memory=True)



    # Get dataset characteristics ...
    args.num_classes = NUM_CLASSES
    args.seq_len = 500
    args.num_channels = 402

    if args.load_from_pretrained:
        best_model_path = pretrain_model(args, resume_checkpoint_path=pretrain_model_path, loss_type="mse")
    else:
        best_model_path = ''

    model, acc_results, f1_results, conf_matrix = train_model(args, best_model_path, resume_checkpoint_path=train_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)
    print("Conf matrix", conf_matrix)

    # append result to a text file...
    text_save_dir = "textFiles"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"Num Classes {NUM_CLASSES}" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results) + "  \n")
    f.write("conf mat:{}".format(conf_matrix) + "  \n")
    f.write("conf mat normed:{}".format(np.round(conf_matrix / np.sum(conf_matrix), 2)) + "  \n")
    f.write('\n')
    f.close()
