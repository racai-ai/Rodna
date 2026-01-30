import os
from pathlib import Path

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
ROOT_EXTRACT_LOG_FILE = os.path.join(data_folder, 'resources', 'root_build.log')
TBL_ROOT2ROOT_FILE = os.path.join(data_folder, 'resources', 'root_rules.ro')
MSD_MAP_FILE = os.path.join(data_folder, 'resources', 'msdtag.ro.map')
MORPHO_MAP_FILE = os.path.join(
    data_folder, 'resources', 'conllu-morpho-features.txt')
SENT_SPLITTER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'splitter')
ROINFLECT_MODEL_FOLDER = os.path.join(data_folder, 'models', 'morphology')
ROINFLECT_CHARID_FILE = os.path.join(data_folder, 'models', 'char_ids.txt')
ROINFLECT_CACHE_FILE = os.path.join(data_folder, 'models', 'unknown_aclasses.txt')
SPLITTER_UNICODE_PROPERTY_FILE = os.path.join(
    data_folder, 'models', 'splitter_unic_props.txt')
SPLITTER_FEAT_LEN_FILE = os.path.join(
    data_folder, 'models', 'splitter_feat_len.txt')
TAGGER_UNICODE_PROPERTY_FILE = os.path.join(
    data_folder, 'models', 'tagger_unic_props.txt')
TAGGER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'tagger')
CLS_TAGGER_MODEL_FOLDER = os.path.join(TAGGER_MODEL_FOLDER, 'cls')
CRF_TAGGER_MODEL_FOLDER = os.path.join(TAGGER_MODEL_FOLDER, 'crf')
PARADIGM_MORPHO_FILE = os.path.join(data_folder, 'resources', 'morphalt.xml')
PARSER_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser')
PARSER_DEPRELS_FILE = os.path.join(
    data_folder, 'resources', 'conllu-deprels.txt')
PARSER1_BERT_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser', 'bert1')
PARSER1_TOKEN_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser', 'tok1')
PARSER2_BERT_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser', 'bert2')
PARSER2_TOKEN_MODEL_FOLDER = os.path.join(data_folder, 'models', 'parser', 'tok2')
