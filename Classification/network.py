

import lightning as L

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy
from timm.layers import DropPath
from timm.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from utils import random_masking_3D



class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, use_adaptive_filter: bool =True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)
        self.use_adaptive_filter: bool = use_adaptive_filter

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).to(torch.float32) - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.use_adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, 
                 dim, 
                 mlp_ratio=3., 
                 drop=0., 
                 drop_path=0., 
                 norm_layer=nn.LayerNorm,
                 use_adaptive_filter: bool = True,
                 use_isb: bool = True,
                 use_asb: bool = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, use_adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.use_isb = use_isb
        self.use_asb = use_asb

    def forward(self, x):
        # Check if both ASB and ICB are true
        if self.use_isb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif self.use_isb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(L.LightningModule):
    def __init__(self, 
                 seq_len, 
                 patch_size, 
                 num_channels, 
                 emb_dim,
                 dropout_rate,
                 depth,
                 num_classes,
                 masking_ratio):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=seq_len, 
            patch_size=patch_size,
            in_chans=num_channels, 
            embed_dim=emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.input_layer = nn.Linear(patch_size, emb_dim)

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=emb_dim, drop=dropout_rate, drop_path=dpr[i])
            for i in range(depth)]
        )

        # Classifier head
        self.head = nn.Linear(emb_dim, num_classes)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        self.masking_ratio = masking_ratio

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=self.masking_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x


class model_pretraining(L.LightningModule):
    def __init__(self, 
                 train_ds_size,
                 loss_type="mse",
                 seq_len=500, 
                 patch_size=16, 
                 num_channels=402, 
                 emb_dim=32,
                 dropout_rate=0.5,
                 depth=5,
                 num_classes=9,
                 masking_ratio=0.4,
                 pretrain_lr=1e-4,
                 pretrain_epochs=100,
                 batch_size=128,):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet(seq_len, 
                 patch_size, 
                 num_channels, 
                 emb_dim,
                 dropout_rate,
                 depth,
                 num_classes,
                 masking_ratio,
)
        self.loss_type = loss_type
        self.pretrain_lr = pretrain_lr
        self.pretrain_epochs = pretrain_epochs
        self.train_ds_size  = train_ds_size 
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.pretrain_lr, weight_decay=1e-4)
        # Example: Cosine annealing scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.pretrain_epochs * (self.train_ds_size // self.batch_size), 
                eta_min=1e-6  # Minimum LR
            ),
            'interval': 'step',  # Update every epoch
            'name': 'lr_scheduler'
        }
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]

        preds, target = self.model.pretrain(data)
        
        if self.loss_type == "mse":
            loss = (preds - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * self.model.mask).sum() / self.model.mask.sum()
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(preds, target, reduction='none', beta=0.01)
            loss = loss.mean(dim=-1)  # same as you did before
            loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")



class model_training(L.LightningModule):
    def __init__(self,
                 train_ds_size,
                 seq_len=500, 
                 patch_size=16, 
                 num_channels=402, 
                 emb_dim=32,
                 dropout_rate=0.5,
                 depth=5,
                 num_classes=9,
                 masking_ratio=0.4,
                 train_lr=1e-4,
                 batch_size=128,
                 num_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet(seq_len, 
                 patch_size, 
                 num_channels, 
                 emb_dim,
                 dropout_rate,
                 depth,
                 num_classes,
                 masking_ratio)
        self.f1 = MulticlassF1Score(num_classes=num_classes)
        self.criterion = LabelSmoothingCrossEntropy()
        self.train_lr = train_lr
        self.train_ds_size  = train_ds_size 
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.train_lr, weight_decay=1e-4)
        
        # Example: Cosine annealing scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.num_epochs * (self.train_ds_size // self.batch_size), 
                eta_min=1e-6  # Minimum LR
            ),
            'interval': 'step',  # Update every epoch
            'name': 'lr_scheduler'
        }
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).to(torch.float32).mean()
        f1 = self.f1(preds, labels)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")