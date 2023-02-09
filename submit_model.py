from datasets.memmap_dataset import MemMapSpatioTemporalDataset
from models.spatiotemporal_model import SpatioTemporalModel
from models.lightning_base import LightningBase
from tqdm.auto import tqdm
import torch
import os

# Training

test_path = "/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2017-testing.nc.bin"
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
    print("No checkpoint found")
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

model = LightningBase.load_from_checkpoint(
    model=model,
    checkpoint_path=ckpt
)

################################################################################

fh_submission = open(f"submissions/{experiment_name}.csv", "w")
fh_submission.write("Id,Predicted\n")

t_offset = 0
step_days = 10
num_days_predict = 365
num_days_test = test_dataset.ntimes

submission_offset = 0
chunk_size = batch_size * num_days_predict

for idx, batch in tqdm(enumerate(test_loader)):
    positions, features = batch
    positions = positions.to('cuda' if use_cuda else 'cpu')
    features = features.to('cuda' if use_cuda else 'cpu')

    with torch.no_grad():
        predictions = model(positions, features)

    predictions = predictions.view(-1)

    yearcut_indices = list(range(0, chunk_size + t_offset, num_days_test))

    subdays_indices = [
        y + k
        for y in yearcut_indices
        for k in range(0, num_days_test, step_days)
    ]

    subdays_indices = list(map(lambda i: i - t_offset, subdays_indices))

    subdays_indices = [
        k
        for k in subdays_indices
        if 0 <= k < min(chunk_size, predictions.shape[0])
    ]

    t_offset = chunk_size - (yearcut_indices[-1] - t_offset)

    predictions_list = predictions[subdays_indices].tolist()

    submission_part = "\n".join(
        [
            f"{i+submission_offset},{pred}"
            for i, pred in enumerate(predictions_list)
        ]
    )

    fh_submission.write(submission_part + "\n")

    submission_offset += len(predictions_list)

fh_submission.close()
