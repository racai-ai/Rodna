import os
from pathlib import Path
__version__ = "1.0.0"

from typing import Set
import torch
import logging
from random import seed

# Get same results from the random number generator
seed(1234)
torch.manual_seed(1234)

# Enable logging.DEBUG for more verbose printing
logging.basicConfig(
    level=logging.INFO,
    datefmt="%d-%m-%Y %H:%M:%S",
    format="%(asctime)s %(levelname)s in %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger('rodna')

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_logged_errors: Set[str] = set()


def log_once(message: str, calling_fn: str, log_level: int = logging.INFO) -> None:
    """Print a given message only once."""

    if message not in _logged_errors:
        logger.log(level=log_level, msg=f"{calling_fn}: {message}")
        _logged_errors.add(message)
    # end if


# Setting data folder...
if os.path.exists('data'):
    data_folder = 'data'
else:
    # In pip mode
    rodna_user_folder = os.path.join(str(Path.home()), '.rodna')

    if not os.path.exists(rodna_user_folder):
        os.mkdir(rodna_user_folder)
    # end if

    data_folder = os.path.join(str(Path.home()), '.rodna', 'data')

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    # end if
# end if


TBL_WORDFORM_FILE = os.path.join(data_folder, 'resources', 'tbl.wordform.ro')
TBL_WORDROOT_FILE = os.path.join(data_folder, 'resources', 'tbl.wordroot.ro')
ROOT_EXTRACT_LOG_FILE = os.path.join(
    data_folder, 'resources', 'root_build.log')
TBL_ROOT2ROOT_FILE = os.path.join(data_folder, 'resources', 'root_rules.ro')
MSD_MAP_FILE = os.path.join(data_folder, 'resources', 'msdtag.ro.map')
MORPHO_MAP_FILE = os.path.join(
    data_folder, 'resources', 'conllu-morpho-features.txt')
SENT_SPLITTER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'splitter')
ROINFLECT_MODEL_FOLDER = os.path.join(data_folder, 'models', 'morphology')
ROINFLECT_CHARID_FILE = os.path.join(data_folder, 'models', 'char_ids.txt')
ROINFLECT_CACHE_FILE = os.path.join(
    data_folder, 'models', 'unknown_aclasses.txt')
SPLITTER_UNICODE_PROPERTY_FILE = os.path.join(
    data_folder, 'models', 'splitter_unic_props.txt')
SPLITTER_FEAT_LEN_FILE = os.path.join(
    data_folder, 'models', 'splitter_feat_len.txt')
TAGGER_UNICODE_PROPERTY_FILE = os.path.join(
    data_folder, 'models', 'tagger_unic_props.txt')
TAGGER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'tagger')
CLS_TAGGER_MODEL_FOLDER = os.path.join(TAGGER_MODEL_FOLDER, 'cls')
CRF_TAGGER_MODEL_FOLDER = os.path.join(TAGGER_MODEL_FOLDER, 'crf')
BERT_FOR_CLS_TAGGER_FOLDER = os.path.join(TAGGER_MODEL_FOLDER, 'cls_bert')
BERT_FOR_CRF_TAGGER_FOLDER = os.path.join(TAGGER_MODEL_FOLDER, 'crf_bert')
PARADIGM_MORPHO_FILE = os.path.join(data_folder, 'resources', 'morphalt.xml')
PARSER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser')
PARSER_DEPRELS_FILE = os.path.join(
    data_folder, 'resources', 'conllu-deprels.txt')
PARSER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser')
PARSER1_BERT_MODEL_FOLDER = os.path.join(PARSER_MODEL_FOLDER, 'bert1')
PARSER2_BERT_MODEL_FOLDER = os.path.join(PARSER_MODEL_FOLDER, 'bert2')


def load_rodna() -> None:
    pass