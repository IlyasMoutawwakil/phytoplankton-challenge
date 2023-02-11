import os
import argparse

import torch
import pytorch_lightning as pl

from datasets.memmap_dataset import MemMapSpatioTemporalDataset
from models.spatiotemporal_model import SpatioTemporalModel
from models.lightning_base import LightningBase
from utils import create_submission


def train(args):
    # Training

    use_cuda = torch.cuda.is_available()
    train_path = args.train_path
    batch_size = args.batch_size
    num_workers = args.nthreads
    max_epochs = args.num_epochs

    # 0 means model will be validating but on data points that it might see during training
    valid_interval_length = args.valid_interval_length
    # 365 days as the interval length for training
    # because randomizing sequences length is overkill
    # when lstm is not used, it's actually a factor by
    # which the batch size is multiplied (batch_size * seq_len)
    train_interval_length = args.train_interval_length

    # Experiment

    num_cnn_layers = args.num_cnn_layers
    num_lstm_layers = args.num_lstm_layers
    num_fc_layers = args.num_fc_layers
    hidden_size = args.hidden_size

    # defines the dimensions of the neighborhood
    # a neighborhood of size 1 means that the axis is not considered for spatial features
    lat_neghborhood_size = args.lat_neighborhood_size
    lon_neghborhood_size = args.lon_neighborhood_size
    depth_neghborhood_size = args.depth_neighborhood_size

    with_position_embedding = bool(args.with_position_embedding)
    with_position_embedding_str = 'with_position_embedding' if with_position_embedding else 'without_position_embedding'

    experiment_name = "spatio_temporal_model_" + \
        str(num_cnn_layers) + "_" +\
        str(num_lstm_layers) + "_" +\
        str(num_fc_layers) + "_" + \
        str(hidden_size) + "_" + \
        str(lat_neghborhood_size) + "_" +\
        str(lon_neghborhood_size) + "_" +\
        str(depth_neghborhood_size) + "_" +\
        with_position_embedding_str

    ckpt = args.resume_from_checkpoint

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
        max_epochs=max_epochs,
        callbacks=[ckpt_callback, prog_bar],
        logger=logger,
    )

    trainer.fit(
        lightning_model,
        train_loader,
        valid_loader,
        ckpt_path=ckpt,
    )


def test(args):

    use_cuda = torch.cuda.is_available()
    test_path = args.test_path
    step_days = args.step_days
    batch_size = args.batch_size
    num_workers = args.num_epochs

    # Model Architecture

    num_cnn_layers = args.num_cnn_layers
    num_lstm_layers = args.num_lstm_layers
    num_fc_layers = args.num_fc_layers
    hidden_size = args.hidden_size

    # defines the dimensions of the neighborhood
    # a neghborhood of size 1 means that the axis is not considered.
    # preferably, the neighborhood size should be odd
    lat_neghborhood_size = args.lat_neighborhood_size
    lon_neghborhood_size = args.lon_neighborhood_size
    depth_neghborhood_size = args.depth_neighborhood_size

    with_position_embedding = bool(args.with_position_embedding)
    with_position_embedding_str = 'with_position_embedding' if with_position_embedding else 'without_position_embedding'

    experiment_name = "spatio_temporal_model_" + \
        str(num_cnn_layers) + "_" +\
        str(num_lstm_layers) + "_" +\
        str(num_fc_layers) + "_" + \
        str(hidden_size) + "_" + \
        str(lat_neghborhood_size) + "_" +\
        str(lon_neghborhood_size) + "_" +\
        str(depth_neghborhood_size) + "_" +\
        with_position_embedding_str

    ckpt = args.modelpath

    test_dataset = MemMapSpatioTemporalDataset(
        bin_path=test_path,
        train=False,
        valid=False,
        lat_neghborhood_size=lat_neghborhood_size,
        lon_neghborhood_size=lon_neghborhood_size,
        depth_neghborhood_size=depth_neghborhood_size,
        overwrite_index=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = SpatioTemporalModel(
        features_size=test_dataset.nfeatures,
        hidden_size=hidden_size,
        targets_size=test_dataset.ntargets,

        num_cnn_layers=num_cnn_layers,
        num_lstm_layers=num_lstm_layers,
        num_fc_layers=num_fc_layers,

        lat_neghborhood_size=lat_neghborhood_size,
        lon_neghborhood_size=lon_neghborhood_size,
        depth_neghborhood_size=depth_neghborhood_size,

        with_position_embedding=with_position_embedding,
        lat_size=test_dataset.nlatitudes,
        lon_size=test_dataset.nlongitudes,
        depth_size=test_dataset.ndepths,
    ).to('cuda' if use_cuda else 'cpu').eval()

    lightning_model = LightningBase.load_from_checkpoint(
        model=model,
        checkpoint_path=ckpt
    ).to('cuda' if use_cuda else 'cpu').eval()

    create_submission(experiment_name,
                      lightning_model,
                      test_loader,
                      step_days)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])

    # Data parameters
    parser.add_argument(
        "--train_path",
        type=str,
        help="The path of train data (.nc.bin)",
        default="/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use " "for loading the data",
        default=7,
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="The number of epochs to train for",
        default=200
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The size of a minibatch",
        default=64
    )
    # parser.add_argument(
    #     "--base_lr",
    #     type=float,
    #     help="The initial learning rate to use",
    #     default=0.0005
    # )
    parser.add_argument(
        "--train_interval_length",
        type=int,
        help="The interval length (number of days) to use for training",
        default=365
    )
    parser.add_argument(
        "--valid_interval_length",
        type=int,
        help="The interval length (number of days) to use for validation (365 means 2016 is used for validation, 0 means training on all the data)",
        default=0
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="The path of the ckpt file to resume from",
        default=None
    )

    # Model Architecture
    parser.add_argument(
        "--num_cnn_layers",
        type=int,
        help="The number of CNN layers for the spatiotemporal model",
        default=3,
    )
    parser.add_argument(
        "--num_lstm_layers",
        type=int,
        help="The number of bidirectional LSTM layers for the spatiotemporal model",
        default=2,
    )
    parser.add_argument(
        "--num_fc_layers",
        type=int,
        help="The number of fully connected layers after the LSTM",
        default=4,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="The hidden size of both CNN and LSTM",
        default=128
    )
    parser.add_argument(
        "--with_position_embedding",
        type=int,
        choices=(0, 1),
        help="To use position embedding",
        default=1
    )

    # Neighborhood Configuration
    parser.add_argument(
        "--lat_neighborhood_size",
        type=int,
        help="The latitude neighborhood size",
        default=1,
    )
    parser.add_argument(
        "--lon_neighborhood_size",
        type=int,
        help="The longitude neighborhood size",
        default=1,
    )
    parser.add_argument(
        "--depth_neighborhood_size",
        type=int,
        help="The depth neighborhood size",
        default=9,
    )

    # # Regularization
    # parser.add_argument(
    #     "--dropout",
    #     type=float,
    #     help="The probability of zeroing before the FC layers",
    #     default=0.5,
    # )

    # For the testing
    parser.add_argument(
        "--test_path",
        type=str,
        help="The path of test data (.nc.bin)",
        default="/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2017-testing.nc.bin",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        help="The path to the ckpt file of the model to load to submit predictions",
        default=None,
    )
    parser.add_argument(
        "--step_days",
        type=int,
        help="The step in days for the submission",
        default=10,
    )

    args = parser.parse_args()

    eval(f"{args.command}(args)")
