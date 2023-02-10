from torchmetrics import MeanSquaredLogError
import pytorch_lightning as pl
import torch

use_cuda = torch.cuda.is_available()
MLSE = MeanSquaredLogError().to('cuda' if use_cuda else 'cpu')


def square_root_mean_squared_log_error_loss(x, y):
    """
    Compute the square root of the mean squared log error loss.

    Parameters
    ----------

    x : torch.Tensor
        The predictions.

    y : torch.Tensor
        The targets.

    Returns
    -------
    torch.Tensor
        The square root of the mean squared log error loss.
    """
    return torch.sqrt(MLSE(x, y))


class LightningBase(pl.LightningModule):
    """
    Base class for PyTorch Lightning models.

    This class is used to wrap a PyTorch model and add
    the necessary methods for PyTorch Lightning to work.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to wrap.
    """

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, positions, features):
        return self.model(positions, features)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        positions, features, targets = batch
        predictions = self(positions, features)

        srmsle_loss = square_root_mean_squared_log_error_loss(
            predictions, targets)

        self.log('train_srmsle', srmsle_loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)

        return srmsle_loss

    def validation_step(self, batch, batch_idx):
        positions, features, targets = batch
        predictions = self(positions, features)

        srmsle_loss = square_root_mean_squared_log_error_loss(
            predictions, targets)

        self.log('valid_srmsle', srmsle_loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
