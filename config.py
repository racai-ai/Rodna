import os

TBL_WORDFORM_FILE = os.path.join('data', 'resources', 'tbl.wordform.ro')
TBL_WORDROOT_FILE = os.path.join('data', 'resources', 'tbl.wordroot.ro')
ROOT_EXTRACT_LOG_FILE = os.path.join('data', 'resources', 'root_build.log')
TBL_ROOT2ROOT_FILE = os.path.join('data', 'resources', 'root_rules.ro')
MSD_MAP_FILE = os.path.join('data', 'resources', 'msdtag.ro.map')
# CoRoLa vectors
EXTERNAL_WORD_EMBEDDINGS_FILE = os.path.join('data', 'resources', 'corola.200.5.vec.gz')
EMBEDDING_VOCABULARY_FILE = os.path.join('data', 'models', 'word_ids.txt')
# FastText vectors
# WORD_EMBEDDINGS_FILE = os.path.join('data', 'resources', 'cc.ro.300.vec.gz')
SENT_SPLITTER_MODEL_FOLDER = os.path.join('data', 'models', 'splitter')
ROINFLECT_MODEL_FOLDER = os.path.join('data', 'models', 'morphology')
ROINFLECT_CHARID_FILE = os.path.join('data', 'models', 'char_ids.txt')
ROINFLECT_CACHE_FILE = os.path.join('data', 'models', 'unknown_aclasses.txt')
SPLITTER_UNICODE_PROPERTY_FILE = os.path.join('data', 'models', 'splitter_unic_props.txt')
SPLITTER_FEAT_LEN_FILE = os.path.join('data', 'models', 'splitter_feat_len.txt')
TAGGER_UNICODE_PROPERTY_FILE = os.path.join('data', 'models', 'tagger_unic_props.txt')
CLS_TAGGER_MODEL_FOLDER = os.path.join('data', 'models', 'tagger', 'cls')
CRF_TAGGER_MODEL_FOLDER = os.path.join('data', 'models', 'tagger', 'crf')
PARADIGM_MORPHO_FILE = os.path.join('data', 'resources', 'morphalt.xml')
