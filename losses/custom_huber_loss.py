"""Module for custom huber loss."""
import torch


class CustomHuberLoss(torch.nn.Module):
    """Custom implementation of the Huber loss function, which is less sensitive to outliers."""
    def __init__(self, delta):
        """
        Initialize the CustomHuberLoss with a delta value.

        :param delta: Threshold at which to switch between quadratic and linear loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        """
        Compute the forward pass of the Custom Huber Loss.

        :param y_true: Ground truth (actual) values.
        :param y_pred: Predicted values.
        :return: Calculated Huber loss.
        """
        error = y_true - y_pred
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * error**2
        large_error_loss = ((self.delta**2 / 2) +
                            self.delta**2 * torch.log((torch.abs(error) + (1 - self.delta))))
        return torch.where(is_small_error, small_error_loss, large_error_loss)
