import os

TBL_WORDFORM_FILE = os.path.join('data', 'resources', 'tbl.wordform.ro')
MSD_MAP_FILE = os.path.join('data', 'resources', 'msdtag.ro.map')
# CoRoLa vectors
WORD_EMBEDDINGS_FILE = os.path.join('data', 'resources', 'corola.200.5.vec.gz')
# FastText vectors
# WORD_EMBEDDINGS_FILE = os.path.join('data', 'resources', 'cc.ro.300.vec.gz')
SENT_SPLITTER_MODEL_FOLDER = os.path.join('data', 'models', 'splitter')
ROINFLECT_MODEL_FOLDER = os.path.join('data', 'models', 'morphology')
ROINFLECT_CHARID_FILE = os.path.join('data', 'models', 'char_ids.txt')
UNICODE_PROPERTY_FILE = os.path.join('data', 'models', 'unic_props.txt')
PREDICTED_AMB_CLASSES_FILE = os.path.join(
    'data', 'resources', 'predicted-ambiguity-classes.txt')
