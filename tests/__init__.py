from rodna.tokenizer import RoTokenizer
from rodna.splitter import RoSentenceSplitter
from rodna.morphology import RoInflect
from rodna.lemmatization import RoLemmatizer
from utils.Lex import Lex

lexicon = Lex()
tokenizer = RoTokenizer(lexicon)
splitter = RoSentenceSplitter(lexicon, tokenizer)
splitter.load()
morphology = RoInflect(lexicon)
morphology.load()
lemmatizer = RoLemmatizer(lexicon, morphology)
