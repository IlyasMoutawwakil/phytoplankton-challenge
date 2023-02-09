import torch


class SpatioTemporalModel(torch.nn.Module):

    def __init__(self,
                 features_size=14,
                 hidden_size=128,
                 targets_size=1,

                 num_lstm_layers=1,
                 num_cnn_layers=1,
                 num_fc_layers=1,

                 lat_neghborhood_size=3,
                 lon_neghborhood_size=3,
                 depth_neghborhood_size=3,

                 with_position_embedding=False,
                 lat_size=100,
                 lon_size=100,
                 depth_size=100,
                 ):

        super().__init__()

        self.features_size = features_size
        self.hidden_size = hidden_size
        self.targets_size = targets_size

        self.num_lstm_layers = num_lstm_layers
        self.num_cnn_layers = num_cnn_layers
        self.num_fc_layers = num_fc_layers

        self.lat_neghborhood_size = lat_neghborhood_size
        self.lon_neghborhood_size = lon_neghborhood_size
        self.depth_neghborhood_size = depth_neghborhood_size

        self.with_position_embedding = with_position_embedding

        in_size = self.features_size
        out_size = self.features_size

        for i in range(self.num_cnn_layers):

            out_size = self.hidden_size

            conv = torch.nn.Conv3d(
                in_channels=in_size,
                out_channels=out_size,
                kernel_size=(min(self.lat_neghborhood_size, 2*(i+1) + 1),
                             min(self.lon_neghborhood_size, 2*(i+1) + 1),
                             min(self.depth_neghborhood_size, 2*(i+1) + 1))
                if i < self.num_cnn_layers - 1
                else (self.lat_neghborhood_size,
                      self.lon_neghborhood_size,
                      self.depth_neghborhood_size),
                # works as a convolutional pooler in the last layer
                padding='same' if i < self.num_cnn_layers - 1 else 0,
                bias=False,
            )
            setattr(self, f'conv_{i}', conv)

            norm = torch.nn.BatchNorm3d(
                self.hidden_size,
                affine=True,
            )
            setattr(self, f'conv_norm_{i}', norm)

            in_size = out_size

        if self.num_lstm_layers > 0:

            out_size = self.hidden_size

            self.lstm = torch.nn.LSTM(
                in_size,
                out_size,
                self.num_lstm_layers,
                batch_first=True,
                bidirectional=True,
                bias=True)

            in_size = 2 * out_size
            out_size = 2 * out_size

        if self.with_position_embedding:
            self.lat_embedding = torch.nn.Embedding(
                lat_size,
                in_size,
            )
            self.lon_embedding = torch.nn.Embedding(
                lon_size,
                in_size
            )
            self.depth_embedding = torch.nn.Embedding(
                depth_size,
                in_size
            )

        for i in range(self.num_fc_layers):
            in_size = in_size if i == 0 else out_size
            out_size = self.targets_size if i == self.num_fc_layers - 1 else out_size // 2

            fc = torch.nn.Linear(
                in_size,
                out_size,
                bias=True
            )

            setattr(self, f'fc_{i}', fc)

    def forward(self, positions, features):

        batch_size = positions.size(0)
        channels = self.features_size

        # from : [batch_size, lat_size, lon_size, depth_size, sequence_length, features_size]
        # to   : [batch_size * sequence_length, lat_size, lon_size, depth_size, features_size]
        x = features.permute(0, -2, -1, 1, 2, 3).flatten(0, 1)

        for i in range(self.num_cnn_layers):
            x = getattr(self, f'conv_{i}')(x)
            x = getattr(self, f'conv_norm_{i}')(x)
            x = torch.relu(x)

            channels = self.hidden_size

        # from : [batch_size * sequence_length, 1, 1, 1, channels]
        # to   : [batch_size, sequence_length, channels]
        x = x.view(batch_size, -1, channels)

        if self.num_lstm_layers > 0:
            x, _ = self.lstm(x)

        if self.with_position_embedding:
            lat_embed = self.lat_embedding(positions[:, 0])[:, None, :]
            lon_embed = self.lon_embedding(positions[:, 1])[:, None, :]
            depth_embed = self.depth_embedding(positions[:, 2])[:, None, :]
            x = x + lat_embed + lon_embed + depth_embed
            x = torch.relu(x)

        for i in range(self.num_fc_layers):
            x = getattr(self, f'fc_{i}')(x)
            x = torch.relu(x)

        return x


if __name__ == '__main__':
    model = SpatioTemporalModel(
        features_size=14,
        hidden_size=128,
        targets_size=1,

        num_cnn_layers=2,
        num_lstm_layers=2,
        num_fc_layers=2,

        lat_neghborhood_size=1,
        lon_neghborhood_size=1,
        depth_neghborhood_size=7,

        with_position_embedding=True,
        lat_size=100,
        lon_size=100,
        depth_size=100,
    )

    print(model)

    positions = torch.ones(8, 3, dtype=torch.int)

    features = torch.randn(8, 1, 1, 7, 365, 14)

    targets = torch.randn(8, 365, 1)

    output = model(positions, features)

    print(output.size())
    print(output.size())
