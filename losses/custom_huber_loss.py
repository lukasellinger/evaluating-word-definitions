import torch


class CustomHuberLoss(torch.nn.Module):
    def __init__(self, delta):
        super(CustomHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * error**2
        large_error_loss = (self.delta**2 / 2) + self.delta**2 * torch.log((torch.abs(error) + (1 - self.delta)))
        return torch.where(is_small_error, small_error_loss, large_error_loss)
