from urllib.parse import urlencode, urljoin
import re
import http.cookiejar
__version__ = "1.0.0"

import os
from pathlib import Path
from typing import Set
import torch
import logging
from random import seed
import zipfile
from tqdm import tqdm
import urllib.request
import tempfile
import shutil


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


def _clear_resource_folder():
    for entry in os.listdir(data_folder):
        full_path = os.path.join(data_folder, entry)

        if os.path.isfile(full_path):
            logger.info(f'Removing file [{full_path}]')
            os.remove(full_path)
        elif os.path.isdir(full_path):
            logger.info(f'Removing folder [{full_path}]')
            shutil.rmtree(full_path)
        # end if
    # end for


def download_large_gdrive_file(file_id: str, destination: str):
    # First URL that triggers the virus warning page
    google_drive_url = "https://drive.google.com/uc?export=download"

    # Cookie jar so Google can track the session
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj))

    # 1) Load the warning page
    resp = opener.open(f"{google_drive_url}&id={file_id}")
    html = resp.read().decode("utf-8")

    # 2) Extract the form action and hidden inputs
    # action="https://drive.usercontent.google.com/download"
    action_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    
    if not action_match:
        raise RuntimeError("Could not find download form action in Google returned HTML")
    # end if

    action_url = action_match.group(1)

    # Extract hidden inputs: name="..." value="..."
    inputs = dict(re.findall(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html))

    # Sanity check: we expect at least id, export, confirm
    if "id" not in inputs:
        inputs["id"] = file_id
    # end if

    if "export" not in inputs:
        inputs["export"] = "download"
    # end if

    # 3) Build the final download URL with query params
    query = urlencode(inputs)
    download_url = action_url

    if "?" in download_url:
        download_url += "&" + query
    else:
        download_url += "?" + query
    # end if

    # 4) Download with progress bar
    response = opener.open(download_url)

    total_size = int(response.headers.get("Content-Length", 0))
    block_size = 8192

    progress = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(destination, "wb") as f:
        while True:
            chunk = response.read(block_size)

            if not chunk:
                break
            # end if

            f.write(chunk)
            progress.update(len(chunk))
        # end while
    # end while

    progress.close()

    if total_size != 0 and progress.n != total_size:
        raise RuntimeError("Error: download size mismatch when downloading Rodna resources")
    # end if


def download_resources():
    # 0. Only if data folder is in user's home directory!
    if data_folder == 'data' or '.rodna' not in data_folder:
        logger.info(f'Using local repository resources for Rodna version {__version__}')
        return
    # end if
    
    # 1. Check current version of the resources
    rodna_version_file = os.path.join(data_folder, 'version.txt')
    to_be_downloaded = True

    if os.path.isfile(rodna_version_file):
        with open(rodna_version_file, mode='r') as f:
            data_folder_version = f.readline().strip()
        # end with

        if data_folder_version == __version__:
            to_be_downloaded = False
        # end if
    # end if

    if to_be_downloaded:
        logger.info(f'Installing resources for Rodna version {__version__}')

        # 1. Clear data folder
        _clear_resource_folder()

        # 2. Download resources to temp folder
        rodna_resource_file = os.path.join(tempfile.gettempdir(), f'rodna-resources-{__version__}.zip')
        google_drive_resource_file_id = '1PHiKHR9QkvGge7HRIFyGNK_Uc0I1xhtd'
        download_large_gdrive_file(file_id=google_drive_resource_file_id,
                                   destination=rodna_resource_file)

        # 3. Unzip it in the data_folder
        with zipfile.ZipFile(rodna_resource_file, mode='r') as zip_ref:
            zip_ref.extractall(data_folder)
        # end with

        # 4. Write version.txt file
        with open(rodna_version_file, mode='w') as f:
            print(__version__, file=f)
        # end with

        logger.info(f'Rodna resources installed in [{data_folder}]')
    else:
        logger.info(f'Resources already installed for Rodna version {__version__}')
        logger.info(f'Installation folder is [{data_folder}]')
    # end if
