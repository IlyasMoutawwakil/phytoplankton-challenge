# phytoplankton-prediction

Our winning code solution for the phytoplankton kaggle competition.

## Structure

The code is structured as follows:


- `data/` contains data files from which data can be loaded. It will be created and populated with the data files and indexes from the competition. The data files are not included in this repository because of their size and (probably) their licence.

- `datasets/` contains the following dataset classes:

  - `bin_dataset.py` contains the `BinDataset` class, which is a dataset that loads data from the binary data file lazily using a binary index. This dataset could not be extended to the spatio-temporal case, because it is not possible to search the binary index file for a specific data point (specifically neghiborhood points which might not be valid and thus not exist in the binary file).
  
  - `array_dataset.py` contains the `ArraySpatioTemporalDataset` class, which is a dataset that loads data from the binary data file and stores it in an `np.array`. This dataset is used for the spatio-temporal case and can only be used with a number of workers equal to 0 (no multiprocessing).

  - `memmap_dataset.py` contains the `MemmapSpatioTemporalDataset` class, which is a dataset that loads data from the binary data file and stores it in an `np.memmap`. This dataset is used for the spatio-temporal case and can be accelerated by using multiple workers depending on the number of cores available.

  - `battle_of_datasets.py` contains a test script that verifies the conformity of outputs and compares the performance of the three datasets.

- `models/` contains the following model classes:

  - `lightning_base.py` contains the `LightningBase` class, which is a wrapper class for a `torch.nn.Module` model. It contains the training and validation methods, which are used to train, validate and log metrics. It also contains the `square_root_mean_squared_log_error_loss` function, which is the loss function used in the competition.

  - `spatio_temporal.py` contains the `SpatioTemporalModel` class, which is a model that uses a spatio-temporal approach using stacks of *3D Convolutions* which extract features from a 3D neighborhood taking into account the spacial dependancy, *Bidirectional Recurrent Neural Networks* (Bi-LSTM) which take into account the temporal dependancy and *FullyConnected* layers which pool the final representation into one value.

- `logs/` contains the tensorboard logs for the training of the model. It will be created when training the model.

- `checkpoints/` contains the checkpoints of the model. It will be created when training the model.

- `submission/` contains the submission file. It will be created in when testing.

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

Then, download the data files from the competition and extract them in the `data/` directory. The data files are not included in this repository because of their size and (probably) their licence.

To train the model, run:

```bash
python main.py --train
```

To create a submission file, run:

```bash
python main.py --test
```

To train and validate the model, run:

```bash
python main.py --train --test
```

When training, the model will be saved in the `checkpoints/` directory and the tensorboard logs will be saved in the `logs/` directory.

Launch tensorboard to visualize the training metrics:

```bash
tensorboard --logdir logs/
```

## Results

Our winning model had a configuration : 1x1x7 neighborhood, 3 CNNs, 2 LSTMs, 4 FCs, with position embedding.

It was trained for 200 epochs with a batch size of 128. The model was trained on a single NVIDIA GeForce RTX 2080 Ti GPU which took less than 2 hours.

The model achieved a score of 0.015 on the private leaderboard and 0.014 on the public leaderboard.