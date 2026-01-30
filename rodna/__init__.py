from typing import Set
import torch
import logging
from random import seed

seed(1234)
torch.manual_seed(1234)

# Enable logging.DEBUG for more verbose printing
logging.basicConfig(
    level=logging.INFO,
    datefmt="%d-%m-%Y %H:%M:%S",
    format="%(asctime)s %(levelname)s in %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger('rodna')

torch.manual_seed(1234)

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_logged_errors: Set[str] = set()


def log_once(message: str, calling_fn: str, log_level: int = logging.INFO) -> None:
    """Print a given message only once."""

    if message not in _logged_errors:
        logger.log(level=log_level, msg=f"{calling_fn}: {message}")
        _logged_errors.add(message)
    # end if
