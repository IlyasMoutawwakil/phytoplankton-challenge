from datasets.memmap_dataset import MemMapSpatioTemporalDataset
from models.spatiotemporal_model import SpatioTemporalModel
from models.lightning_base import LightningBase
import pytorch_lightning as pl
import torch
import os

# Training

train_path = "/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin"
use_cuda = torch.cuda.is_available()
batch_size = 128
num_workers = 8

# 0 means model will be validating but on data points that it might see during training
valid_interval_length = 0
# 365 days as the interval length for training
# because randomizing sequences length is overkill
# when lstm is not used, it's actually a factor by
# which the batch size is multiplied (batch_size * seq_len)
train_interval_length = 365

# Experiment

num_cnn_layers = 3
num_lstm_layers = 2
num_fc_layers = 4
hidden_size = 128

# defines the dimensions of the neighborhood
# a neghborhood of size 1 means that the axis is not considered for spatial features
lat_neghborhood_size = 1
lon_neghborhood_size = 1
depth_neghborhood_size = 11

with_position_embedding = True

experiment_name = f"spatio_temporal_model_\
{num_cnn_layers}_\
{num_lstm_layers}_\
{num_fc_layers}_\
{hidden_size}_\
{lat_neghborhood_size}_\
{lon_neghborhood_size}_\
{depth_neghborhood_size}_\
{'with_position_embedding' if with_position_embedding else 'without_position_embedding'}"

try:
    os.listdir(f"checkpoints/{experiment_name}").index("last.ckpt")
    ckpt = f"checkpoints/{experiment_name}/last.ckpt"
    print(f"Resuming from checkpoint: {ckpt}")

except Exception:
    ckpt = None


# Data

train_dataset = MemMapSpatioTemporalDataset(
    bin_path=train_path,
    train=True,
    valid=False,
    valid_interval_length=valid_interval_length,
    train_interval_length=train_interval_length,
    lat_neghborhood_size=lat_neghborhood_size,
    lon_neghborhood_size=lon_neghborhood_size,
    depth_neghborhood_size=depth_neghborhood_size,
    overwrite_index=False,
)

valid_dataset = MemMapSpatioTemporalDataset(
    bin_path=train_path,
    train=True,
    valid=True,
    train_interval_length=train_interval_length,
    lat_neghborhood_size=lat_neghborhood_size,
    lon_neghborhood_size=lon_neghborhood_size,
    depth_neghborhood_size=depth_neghborhood_size,
    overwrite_index=False,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_cuda,
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
)

# Model

model = SpatioTemporalModel(
    features_size=train_dataset.nfeatures,
    hidden_size=hidden_size,
    targets_size=train_dataset.ntargets,

    num_cnn_layers=num_cnn_layers,
    num_lstm_layers=num_lstm_layers,
    num_fc_layers=num_fc_layers,

    lat_neghborhood_size=lat_neghborhood_size,
    lon_neghborhood_size=lon_neghborhood_size,
    depth_neghborhood_size=depth_neghborhood_size,

    with_position_embedding=with_position_embedding,
    lat_size=train_dataset.nlatitudes,
    lon_size=train_dataset.nlongitudes,
    depth_size=train_dataset.ndepths,
).to("cuda" if use_cuda else "cpu")

print(model)

# Unfortunatly, the DCE does not currently support PyTorch 2.0  which requires Cuda 11.7
# model = torch.compile(model)

lightning_model = LightningBase(
    model
).to("cuda" if use_cuda else "cpu")

# Callbacks

prog_bar = pl.callbacks.progress.TQDMProgressBar(
    refresh_rate=1,
)

logger = pl.loggers.TensorBoardLogger(
    save_dir=f"logs/{experiment_name}/",
)

ckpt_callback = pl.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{experiment_name}/",
    filename="checkpoint-{epoch:03d}-{valid_srmsle:.5f}-{train_srmsle:.5f}",
    monitor="valid_srmsle",
    save_last=True,
    save_top_k=3,
    mode="min",
)

# Training

trainer = pl.Trainer(
    accelerator='gpu',
    benchmark=True,
    enable_progress_bar=True,
    log_every_n_steps=1,
    num_sanity_val_steps=1,
    check_val_every_n_epoch=1,
    max_epochs=200,
    callbacks=[ckpt_callback, prog_bar],
    logger=logger,
)

trainer.fit(
    lightning_model,
    train_loader,
    valid_loader,
    ckpt_path=ckpt,
)
