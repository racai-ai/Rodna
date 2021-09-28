from rodna.tokenizer import RoTokenizer
from rodna.splitter import RoSentenceSplitter
from config import TBL_WORDFORM_FILE, WORD_EMBEDDINGS_FILE

tokenizer = RoTokenizer(TBL_WORDFORM_FILE, WORD_EMBEDDINGS_FILE)
splitter = RoSentenceSplitter(tokenizer)
splitter.load_keras_model()
