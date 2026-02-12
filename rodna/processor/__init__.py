import torch
from random import seed

# Get same results from the random number generator
seed(1234)
torch.manual_seed(1234)

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
