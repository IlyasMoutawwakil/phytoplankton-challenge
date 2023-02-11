# Phytoplankton Challenge

Our winning code solution for the phytoplankton kaggle competition.

## Structure & Documentation

The code is structured as follows:


- `data/` contains data files from which data can be loaded. It will be created and populated with the data files and indexes from the competition. The data files are not included in this repository because of their size and (probably) their licence.

- `datasets/` contains the following dataset classes:

  - `bin_dataset.py` contains the `BinDataset` class, which is a dataset that loads data from the binary data file lazily using a binary index. This dataset could not be extended to the spatio-temporal case, because it is not possible to search the binary index file for a specific data point (specifically neghiborhood points which might not be valid and thus not exist in the binary file).
  
  - `array_dataset.py` contains the `ArraySpatioTemporalDataset` class, which is a dataset that loads data from the binary data file and stores it in an `np.array`. This dataset is used for the spatio-temporal case and can only be used with a number of workers equal to 0 (no multiprocessing).

  - `memmap_dataset.py` contains the `MemmapSpatioTemporalDataset` class, which is a dataset that loads data from the binary data file and stores it in an `np.memmap`. This dataset is used for the spatio-temporal case and can be accelerated by using multiple workers depending on the number of cores available.
  
  Datasets return a *random cut* from the time series in the training set (a time series for each point) which helped in augmenting the data, regularizing the model and thus preventing overfitting to some degree.

  - `battle_of_datasets.py` contains a test script that verifies the conformity of outputs and compares the performance of the three datasets.

- `models/` contains the following model classes:

  - `lightning_base.py` contains the `LightningBase` class, which is a wrapper class for a `torch.nn.Module` model. It contains the training and validation methods, which are used to train, validate and log metrics. It also contains the `square_root_mean_squared_log_error_loss` function, which is the loss function used in the competition.

  - `spatio_temporal.py` contains the `SpatioTemporalModel` class, which is a model that uses a spatio-temporal approach using stacks of *3D Convolutions* which extract features from a 3D neighborhood taking into account the spacial dependancy, *Bidirectional Recurrent Neural Networks* (Bi-LSTM) which take into account the temporal dependancy and *FullyConnected* layers which pool the final representation into one value.

- `logs/` contains the tensorboard logs for the training of the model. It will be created when training the model.

- `checkpoints/` contains the checkpoints of the model. It will be created when training the model.

- `submissions/` contains the submission files. It will be created in when testing.

- `debug.ipynb` contains a notebook for debugging the model's predictions.

- `requirements.txt` contains the required python packages for a fully functional DCE environment with no conflicts.

- `utils.py` contains a utility function to create a submission file from a model checkpoint.

- `main.py` contains the main script. It can be used to train, validate, and create a submission file.

## Usage

First, create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
cat requirements.txt | xargs -n 1 pip install
```

if you encounter any problems with the installation of the `torch` package, you can try to install it manually:

```bash
pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116 --force
```

Then, download the data files from the competition and extract them in the `data/` directory or have them somewhere on your machine (PATH_TO_TRAINING_SET, PATH_TO_TEST_SET). The data files are not included in this repository because of their size and (probably) their licence.

To train the model, run:

```bash
python3 main.py train --train_path PATH_TO_TRAINING_SET
```

To create a submission file, run:

```bash
python3 main.py test --modelpath PATH_TO_CHECKPOINT --test_path PATH_TO_TEST_SET 
```

To recieve additional information, run:

```bash
python3 main.py --help
```

The possible arrguments are as follows:
- Possible commands: `train` and `test`.
- Optional arguments:
  - `-h`, `--help`: show help message and exit.
  - `--train_path`: the path to the training set (.nc.bin file), default is `/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin`.
  - `--test_path`: the path to the test set (.nc.bin file), default is `/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2017-testing.nc.bin`.
  - `--modelpath`: the model of the path to use to generate predictions on the test set.
  - `--step_days`: The step in days for the submission, default is 10.
  - `--nthreads`: The number of threads to use for loading the data, default is 7.
  - `--num_epochs`: The number of epochs to train for, default is 200.
  - `--batch_size`: The size of a minibatch", default is 64.
  - `--train_interval_length`: The interval length (number of days) to use for training, default is 365.
  - `--valid_interval_length`: The interval length (number of days) to use for validation (365 means 2016 is used for validation, 0 means training on all the data), default is 0.
  - `--resume_from_checkpoint`: The path of the ckpt file to resume from, default is None.
  - `--num_cnn_layers`: The number of CNN layers for the spatiotemporal model, default is 3.
  - `--num_lstm_layers`: The number of bidirectional LSTM layers for the spatiotemporal model, default is 2.
  - `--num_fc_layers`: The number of fully connected layers after the LSTM, default is 4.
  - `--hidden_size`: The hidden size of both CNN and LSTM, default is 128.
  - `--with_position_embedding`: An integer variable that takes 1 to use positional embedding and 0 to not use it, default is 1.
  - `--lat_neighborhood_size`: The latitude neighborhood size, default is 1.
  - `--lon_neighborhood_size`: The longitude neighborhood size, default is 1.
  - `--depth_neighborhood_size`: The depth neighborhood size, default is 9.


When training, the model will be saved in the `checkpoints/` directory and the tensorboard logs will be saved in the `logs/` directory.

Launch tensorboard to visualize the training metrics:

```bash
tensorboard --logdir logs/
```

## Results

Our winning model had a configuration : 1x1x9 neighborhood, 3 CNNs, 2 LSTMs, 4 FCs, with position embedding.

It was trained for 200 epochs with a batch size of 64. The model was trained on a single NVIDIA GeForce RTX 2080 Ti GPU which took a little more than 2 hours (that was before using `MemmapSpatioTemporalDataset` which accelerated training for even faster experimentation).

Its tensorboad logs are in the `logs/cnn_lstm_3_2_4_128_1_1_9_with_position_embedding/` folder.

The model achieved a score of 0.01500 on the public leaderboard and 0.01494 on the private leaderboard.