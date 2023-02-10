from datasets.memmap_dataset import MemMapSpatioTemporalDataset
from models.spatiotemporal_model import SpatioTemporalModel
from models.lightning_base import LightningBase
from utils import create_submission

import torch
import os


if __name__ == '__main__':

    # Testing

    test_path = "/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2017-testing.nc.bin"
    step_days = 10

    use_cuda = torch.cuda.is_available()
    batch_size = 128
    num_workers = 8

    # Experiment

    num_cnn_layers = 3
    num_lstm_layers = 2
    num_fc_layers = 4
    hidden_size = 128

    # defines the dimensions of the neighborhood
    # a neghborhood of size 1 means that the axis is not considered.
    # preferably, the neighborhood size should be odd
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
        ckpt = os.listdir(f"checkpoints/{experiment_name}")[-1]
        ckpt = f"checkpoints/{experiment_name}/{ckpt}"
        print("Submitting with checkpoint", ckpt)
    except:
        print("No checkpoint with this configuration found.")
        exit()

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
