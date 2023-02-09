# Standard imports
import os
import pathlib
import struct

# External imports
import torch
import numpy as np
from tqdm.auto import tqdm

_ENCODING_INT = "I"
_ENCODING_ENDIAN = "<"

_IN_VARIABLES = [
    "dissic",
    "mlotst",
    "nh4",
    "no3",
    "nppv",
    "o2",
    "ph",
    "po4",
    "so",
    "talk",
    "thetao",
    "uo",
    "vo",
    "zos",
]

_OUT_VARIABLES = [
    "phyc"
]


def write_bin_data(bin_file, fmt, values):
    bin_file.write(struct.pack(fmt, *values))


def read_bin_data(bin_file, offset, whence, fmt):
    with open(bin_file.name, "rb") as f:
        f.seek(offset, whence)
        nbytes = struct.calcsize(fmt)
        values = struct.unpack(fmt, f.read(nbytes))
    return values, nbytes


def read_dim(bin_file, offset, base_format):

    fmt = _ENCODING_ENDIAN + "i"
    (dim, nbytes_dim) = read_bin_data(bin_file, offset, 0, fmt)

    dim = dim[0]  # the returned values is a tuple
    offset += nbytes_dim

    fmt = _ENCODING_ENDIAN + (base_format * dim)
    (values, nbytes_values) = read_bin_data(bin_file, offset, 0, fmt)

    return dim, np.array(values), nbytes_dim + nbytes_values


class BinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bin_path,
        train=True,
        valid=False,
        train_interval_length=365,
        valid_interval_length=365,
        overwrite_index=False,
    ):
        super().__init__()

        self.bin_path = pathlib.Path(bin_path)
        self.bin_file = open(self.bin_path, "rb")

        self.train = train
        self.valid = valid

        self.overwrite_index = overwrite_index

        self.train_interval_length = train_interval_length
        self.valid_interval_length = valid_interval_length

        self.nfeatures = len(_IN_VARIABLES)
        self.ntargets = len(_OUT_VARIABLES)
        self.ncolumns = self.nfeatures + int(self.train)

        self.row_format = "?" + "f" * self.ncolumns
        self.row_size = struct.calcsize(_ENCODING_ENDIAN + self.row_format)

        self.header_offset = 0

        self.nlatitudes, self.latitudes, nbytes = read_dim(
            self.bin_file, self.header_offset, "f"
        )
        self.header_offset += nbytes

        self.nlongitudes, self.longitudes, nbytes = read_dim(
            self.bin_file, self.header_offset, "f"
        )
        self.header_offset += nbytes

        self.ndepths, self.depths, nbytes = read_dim(
            self.bin_file, self.header_offset, "f"
        )
        self.header_offset += nbytes

        self.ntimes, self.times, nbytes = read_dim(
            self.bin_file, self.header_offset, "i"
        )

        self.header_offset += nbytes

        self.lat_chunk_size = self.nlongitudes * self.ndepths * self.ntimes
        self.lon_chunk_size = self.ndepths * self.ntimes
        self.depth_chunk_size = self.ntimes

        self._load_bin_index()

    def _load_bin_index(self):
        index_path = pathlib.Path(f"data/{self.bin_path.name}.bin_index")

        # Generate the index if necessary or requested
        if not index_path.exists() or self.overwrite_index:
            os.makedirs(index_path.parent, exist_ok=True)
            self._generate_index(index_path)

        self._bin_index = open(index_path, "rb")

    def _generate_index(self, index_path):

        bin_index = open(index_path, "wb")

        # Rewind the data file handler to just after the header
        self.bin_file.seek(self.header_offset)

        idx = 0
        for ilatitude in tqdm(range(self.nlatitudes)):
            for ilongitude in range(self.nlongitudes):
                for idepth in range(self.ndepths):

                    data_offset = (
                        self.header_offset
                        + (
                            ilatitude * self.lat_chunk_size
                            + ilongitude * self.lon_chunk_size
                            + idepth * self.depth_chunk_size
                        )
                        * self.row_size
                    )

                    self.bin_file.seek(data_offset)
                    (is_valid,) = struct.unpack(
                        "<?", self.bin_file.read(struct.calcsize("<?"))
                    )

                    if is_valid:
                        fmt = (
                            _ENCODING_ENDIAN + (_ENCODING_INT * 5)
                        )
                        write_bin_data(
                            bin_index,
                            fmt,
                            (
                                idx,
                                data_offset,
                                ilatitude,
                                ilongitude,
                                idepth,
                            ),
                        )

                        idx += 1

        bin_index.close()

    def _get_fileoffset(self, idx):

        whence = 0 if idx >= 0 else 2
        fmt = _ENCODING_ENDIAN + (_ENCODING_INT * 5)
        offset = idx * struct.calcsize(fmt)

        values, _ = read_bin_data(self._bin_index, offset, whence, fmt)

        return values

    def __getitem__(self, idx):
        (_, bin_offset, ilatitude, ilongitude, idepth) = self._get_fileoffset(idx)
        position = [ilatitude, ilongitude, idepth]

        self.bin_file.seek(bin_offset)
        values, _ = read_bin_data(
            self.bin_file, bin_offset, 0,
            _ENCODING_ENDIAN + (self.row_format * self.ntimes),
        )

        values = np.array(values).reshape(self.ntimes, 1+self.ncolumns)[:, 1:]

        if self.train:
            if self.valid:
                # Take the last self.temporal_valid_days
                values = values[-self.valid_interval_length:, :]

            else:
                # Take a random cut of self.train_interval_length days before the last self.temporal_valid_days
                random_cut = np.random.randint(
                    0, self.ntimes - self.train_interval_length - self.valid_interval_length + 1)

                values = values[random_cut: random_cut +
                                self.train_interval_length, :]

            # remove the phyc column
            features = values[:, :-1]
            # get the phyc column
            targets = values[:, -1].reshape(-1, 1)

            features = torch.tensor(features, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)
            positions = torch.tensor(position, dtype=torch.int)

            return positions, features, targets

        else:
            features = torch.tensor(features, dtype=torch.float32)
            positions = torch.tensor(position, dtype=torch.int)

            return positions, features

    def __len__(self):
        values = self._get_fileoffset(-1)
        return values[0] + 1


if __name__ == "__main__":

    train_path = "data/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin"
    dataset = BinDataset(
        train_path,
        train=True,
        valid=False,
        train_interval_length=365,
        valid_interval_length=365,
        overwrite_index=False,
    )
    print(len(dataset))

    for i in tqdm(range(0, len(dataset), 100)):
        position, features, targets = dataset[i]
        # print(position.shape)
        # print(features.shape)
        # print(targets.shape)
    position, features, targets = dataset[len(dataset)]
    # print(position.shape)
    # print(features.shape)
    # print(targets.shape)
