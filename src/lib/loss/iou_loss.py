from torch import nn


class IoULoss(nn.Module):
    reduction: str

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    

    def forward(self, predicted, expected):
        
        
        
    