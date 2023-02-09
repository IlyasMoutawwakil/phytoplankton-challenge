# Standard imports
import os
import pathlib
import struct

# External imports
import torch
import numpy as np
from tqdm.auto import tqdm

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


def read_bin_data(bin_file, fmt):
    nbytes = struct.calcsize(fmt)
    values = struct.unpack(fmt, bin_file.read(nbytes))
    return values, nbytes


def read_dim(bin_file, offset, base_format):
    fmt = _ENCODING_ENDIAN + "i"
    bin_file.seek(offset)
    (dim, nbytes_dim) = read_bin_data(bin_file, fmt)

    dim = dim[0]  # the returned values is a tuple
    offset += nbytes_dim

    fmt = _ENCODING_ENDIAN + (base_format * dim)
    bin_file.seek(offset)
    (values, nbytes_values) = read_bin_data(bin_file, fmt)

    return dim, np.array(values), nbytes_dim + nbytes_values


class ArraySpatioTemporalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bin_path,
        train=True,
        valid=False,
        train_interval_length=365,
        valid_interval_length=365,
        lat_neghborhood_size=1,
        lon_neghborhood_size=1,
        depth_neghborhood_size=1,
        overwrite_index=False,
    ):
        super().__init__()

        self.bin_path = pathlib.Path(bin_path)
        self.bin_file = open(bin_path, "rb")

        self.train = train
        self.valid = valid

        self.overwrite_index = overwrite_index

        self.train_interval_length = train_interval_length
        self.valid_interval_length = valid_interval_length

        self.lat_neghborhood_size = lat_neghborhood_size
        self.lon_neghborhood_size = lon_neghborhood_size
        self.depth_neghborhood_size = depth_neghborhood_size

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

        self._load_arrays()

    def _load_arrays(self):
        index_path = pathlib.Path.cwd(
        ) / f"data/{self.bin_path.stem}.array_index.npy"
        data_path = pathlib.Path.cwd(
        ) / f"data/{self.bin_path.stem}.array_data.npy"

        if not index_path.exists() or not data_path.exists() or self.overwrite_index:
            os.makedirs(index_path.parent, exist_ok=True)
            self._generate_arrays(index_path, data_path)

        self._array_index = np.load(index_path)
        self._array_data = np.load(data_path)

    def _generate_arrays(self, index_path, data_path):
        # Rewind the data file handler to just after the header
        self.bin_file.seek(self.header_offset, 0)

        array_data = np.zeros((self.nlatitudes, self.nlongitudes, self.ndepths,
                               self.ntimes, self.ncolumns), dtype=np.float32)

        index_list = []
        for ilatitude in tqdm(range(self.nlatitudes)):
            for ilongitude in range(self.nlongitudes):
                for idepth in range(self.ndepths):

                    # Compute the offset of the first time sample
                    offset = (self.header_offset + (
                        ilatitude * self.lat_chunk_size + ilongitude *
                        self.lon_chunk_size + idepth * self.depth_chunk_size
                    ) * self.row_size)

                    self.bin_file.seek(offset, 0)
                    (is_valid,) = struct.unpack(
                        "<?", self.bin_file.read(struct.calcsize("<?"))
                    )

                    # If the location is valid we record its index
                    if is_valid:
                        index_list.append([ilatitude, ilongitude, idepth])

                        self.bin_file.seek(offset, 0)
                        values, _ = read_bin_data(
                            self.bin_file,
                            _ENCODING_ENDIAN + (self.row_format * self.ntimes),
                        )
                        array_data[ilatitude,
                                   ilongitude,
                                   idepth,
                                   :, :] = np.array(values).reshape(
                            self.ntimes, 1+self.ncolumns)[:, 1:]

        array_index = np.array(index_list)

        np.save(index_path, array_index)
        np.save(data_path, array_data)

    def __getitem__(self, idx):

        # File offset for the i-th sample
        position = self._array_index[idx]
        ilatitude, ilongitude, idepth = position

        lat_radius = self.lat_neghborhood_size // 2
        lon_radius = self.lon_neghborhood_size // 2
        depth_radius = self.depth_neghborhood_size // 2

        start_lat = max(ilatitude-lat_radius, 0)
        end_lat = min(ilatitude+lat_radius+1, self.nlatitudes)

        start_lon = max(ilongitude-lon_radius, 0)
        end_lon = min(ilongitude+lon_radius+1, self.nlongitudes)

        start_depth = max(idepth-depth_radius, 0)
        end_depth = min(idepth+depth_radius+1, self.ndepths)

        # Create the neighborhood array with self.num_space_dimentions
        neghborhood = np.zeros(
            (self.lat_neghborhood_size,
             self.lon_neghborhood_size,
             self.depth_neghborhood_size,
             self.ntimes, self.ncolumns)
        )

        neghborhood[lat_radius - (ilatitude - start_lat):lat_radius + (end_lat - ilatitude),
                    lon_radius - (ilongitude - start_lon):lon_radius + (end_lon - ilongitude),
                    depth_radius - (idepth - start_depth):depth_radius + (end_depth - idepth),
                    :, :] = self._array_data[start_lat:end_lat,
                                             start_lon:end_lon,
                                             start_depth:end_depth,
                                             :, :]

        if self.train:
            if self.valid:
                # Take the last self.temporal_valid_days
                neghborhood = neghborhood[:, :, :,
                                          -self.valid_interval_length:, :]

            else:
                # Take a random cut of self.train_interval_length days before the last self.temporal_valid_days
                random_cut = np.random.randint(
                    0, self.ntimes - self.train_interval_length - self.valid_interval_length + 1)

                neghborhood = neghborhood[:, :, :,
                                          random_cut: random_cut + self.train_interval_length, :]

            # remove the phyc column
            features = neghborhood[:, :, :, :, :-1]
            # get the phyc column
            targets = neghborhood[lat_radius,
                                  lon_radius,
                                  depth_radius,
                                  :, -1].reshape(-1, 1)

            features = torch.tensor(features, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)
            positions = torch.tensor(position, dtype=torch.int)

            return positions, features, targets

        else:
            features = neghborhood
            features = torch.tensor(features, dtype=torch.float32)
            positions = torch.tensor(position, dtype=torch.int)

            return positions, features

    def __len__(self):
        return len(self._array_index)


if __name__ == "__main__":

    train_path = "data/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin"
    dataset = ArraySpatioTemporalDataset(
        train_path,
        train=True,
        valid=False,
        train_interval_length=365,
        valid_interval_length=365,
        lat_neghborhood_size=5,
        lon_neghborhood_size=5,
        depth_neghborhood_size=5,
        overwrite_index=False,
    )
    print(len(dataset))

    position, features, targets = dataset[0]
    print(position.shape)
    print(features.shape)
    print(targets.shape)
