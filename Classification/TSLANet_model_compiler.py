import argparse
from typing import Any
import os

import lightning as L

# import webdataset as wds
# from pathlib import Path
from streaming import StreamingDataset

import torch
import torch.nn as nn

from utils import str2bool

from classification.network import model_training


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



if __name__ == "__main__":
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

    NUM_CLASSES = 9
    args.num_classes = NUM_CLASSES
    args.seq_len = 500
    args.num_channels = 402


    # Create input on CPU
    dummy_input = torch.randn(1, 402, 500)

    # 1. Load checkpoint and move to target device
    ckpt = "/Users/danielfisher/repositories/TSLANet/lightning_logs/AUDUSD_CLF_dim32_depth5_ASB_True_AF_True_ICB_True_preTr_True_patch_8_batch_128_20250429_130239_train/epoch=43-step=30932.ckpt"

    model = model_training.load_from_checkpoint("/path/to/checkpoint.ckpt")

    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    y_hat = model(x)


