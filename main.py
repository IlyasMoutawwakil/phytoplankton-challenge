import argparse
from utils import train, test

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])

    # Train parameters
    parser.add_argument(
        "--train_path",
        type=str,
        help="The path of train data (.nc.bin)",
        default="/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2010-2016-training.nc.bin",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="The number of threads to use " "for loading the data",
        default=7,
    )
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

    # Testing specific parameters
    parser.add_argument(
        "--test_path",
        type=str,
        help="The path of test data (.nc.bin)",
        default="/mounts/Datasets3/2022-ChallengePlankton/sub_2CMEMS-MEDSEA-2017-testing.nc.bin",
    )
    parser.add_argument(
        "--model_path",
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
