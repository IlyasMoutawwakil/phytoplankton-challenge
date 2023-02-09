from memmap_dataset import MemMapSpatioTemporalDataset
from array_dataset import ArraySpatioTemporalDataset
from bin_dataset import BinDataset

import numpy as np
import torch
import time

train_path = "data/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin"

memmap_dataset = MemMapSpatioTemporalDataset(
    bin_path=train_path,
    train=True,
    valid=True,
    train_interval_length=365,
    valid_interval_length=365,
    lat_neghborhood_size=1,
    lon_neghborhood_size=1,
    depth_neghborhood_size=1,
    overwrite_index=False,
)

array_dataset = ArraySpatioTemporalDataset(
    bin_path=train_path,
    train=True,
    valid=True,
    train_interval_length=365,
    valid_interval_length=365,
    lat_neghborhood_size=1,
    lon_neghborhood_size=1,
    depth_neghborhood_size=1,
    overwrite_index=False,
)

bin_dataset = BinDataset(
    bin_path=train_path,
    train=True,
    valid=True,
    train_interval_length=365,
    valid_interval_length=365,
    overwrite_index=False,
)

rand_idx = np.random.randint(0, len(memmap_dataset))
np.random.seed(0)
memmap_sample = memmap_dataset[rand_idx]
np.random.seed(0)
array_sample = array_dataset[rand_idx]
np.random.seed(0)
bin_sample = bin_dataset[rand_idx]

# Test output is equal
assert np.allclose(memmap_sample[0], array_sample[0])
assert np.allclose(memmap_sample[0], bin_sample[0])
assert np.allclose(memmap_sample[1], array_sample[1])
assert np.allclose(memmap_sample[1], bin_sample[1])
assert np.allclose(memmap_sample[2], array_sample[2])
assert np.allclose(memmap_sample[2], bin_sample[2])

print("Dataset Equality Test Passed")

memmap_loader = torch.utils.data.DataLoader(
    memmap_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

array_loader = torch.utils.data.DataLoader(
    array_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,  # because it's in ram
    pin_memory=True,
)

bin_loader = torch.utils.data.DataLoader(
    bin_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

memmap_iter = iter(memmap_loader)
array_iter = iter(array_loader)
bin_iter = iter(bin_loader)

# warm up
for i in range(10):
    next(memmap_iter)
    next(array_iter)
    next(bin_iter)

# time memmap loader
t0 = time.time()
while True:
    try:
        next(memmap_iter)
    except StopIteration:
        break
t1 = time.time()
print("memmap loader time: ", t1 - t0)

# time array loader
t0 = time.time()
while True:
    try:
        next(array_iter)
    except StopIteration:
        break
t1 = time.time()
print("array loader time: ", t1 - t0)

# time bin loader
t0 = time.time()
while True:
    try:
        next(bin_iter)
    except StopIteration:
        break
t1 = time.time()
print("bin loader time: ", t1 - t0)

print("No neighborhood Benchmark Passed")

memmap_dataset = MemMapSpatioTemporalDataset(
    bin_path=train_path,
    train=True,
    valid=True,
    train_interval_length=365,
    valid_interval_length=365,
    lat_neghborhood_size=3,
    lon_neghborhood_size=3,
    depth_neghborhood_size=3,
    overwrite_index=False,
)

array_dataset = ArraySpatioTemporalDataset(
    bin_path=train_path,
    train=True,
    valid=True,
    train_interval_length=365,
    valid_interval_length=365,
    lat_neghborhood_size=3,
    lon_neghborhood_size=3,
    depth_neghborhood_size=3,
    overwrite_index=False,
)

memmap_loader = torch.utils.data.DataLoader(
    memmap_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

array_loader = torch.utils.data.DataLoader(
    array_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,  # because it's in ram
    pin_memory=True,
)

memmap_iter = iter(memmap_loader)
array_iter = iter(array_loader)

# warm up
for i in range(10):
    next(memmap_iter)
    next(array_iter)

# time memmap loader
t0 = time.time()
while True:
    try:
        next(memmap_iter)
    except StopIteration:
        break
t1 = time.time()
print("memmap loader time: ", t1 - t0)

# time array loader
t0 = time.time()
while True:
    try:
        next(array_iter)
    except StopIteration:
        break
t1 = time.time()
print("array loader time: ", t1 - t0)

print("3x3 neighborhood Benchmark Passed")

# Dataset Equality Test Passed

# memmap loader time:  0.7793374061584473
# array loader time:  0.6964735984802246
# bin loader time:  10.466963529586792
# No neighborhood Benchmark Passed

# memmap loader time:  20.68733835220337
# array loader time:  26.191723108291626
# 3x3x3 neighborhood Benchmark Passed