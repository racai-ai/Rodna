import os
from . import data_folder


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
