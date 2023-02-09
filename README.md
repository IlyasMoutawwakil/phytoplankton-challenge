# phytoplankton-prediction

Our winning code solution for the phytoplankton kaggle competition.

## Structure

The code is structured as follows:


- `data/` contains the data files. Initially, It should contain a binary data files from which data can be loaded.
- `datasets/` contains the dataset classes:
  -  `bin_dataset.py` contains the `BinDataset` class, which is a dataset that loads data from the binary data and using an binary index. This dataset could not be extended to the spatio-temporal case, because it is not possible to search the data binary file for a specific index (specifically neghiborhood indexes which might not exist).
  -  `array_dataset.py` contains the `ArraySpatioTemporalDataset` class, which is a dataset that loads data from a binary and stores it in memory. This dataset is used for the spatio-temporal case.