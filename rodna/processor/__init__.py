import re
import torch
from random import seed

# Get same results from the random number generator
seed(1234)
torch.manual_seed(1234)

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_space_rx = re.compile('\\s+')

def normalize_text(text: str) -> str:
    """Takes the input text and makes sure that:
    - Romanian diacritics are all up-to-date,
    - Spaces are all standard."""

    text = text.strip()
    text = _space_rx.sub(' ', text)
    text = text.replace('ş', 'ș')
    text = text.replace('ţ', 'ț')
    text = text.replace('Ş', 'Ș')
    text = text.replace('Ţ', 'Ț')

    return text
