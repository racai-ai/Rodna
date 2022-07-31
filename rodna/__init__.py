import torch

torch.manual_seed(1234)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
