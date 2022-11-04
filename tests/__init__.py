from rodna.tokenizer import RoTokenizer
from rodna.splitter import RoSentenceSplitter
from rodna.morphology import RoInflect
from rodna.lemmatization import RoLemmatizer
from rodna.tagger import RoPOSTagger
from rodna.parser import RoDepParser
from utils.Lex import Lex

lexicon = Lex()
tokenizer = RoTokenizer(lexicon)
splitter = RoSentenceSplitter(lexicon, tokenizer)
splitter.load()
morphology = RoInflect(lexicon)
morphology.load()
tagger = RoPOSTagger(lexicon, tokenizer, morphology, splitter)
tagger.load()
lemmatizer = RoLemmatizer(lexicon, morphology)
parser = RoDepParser(msd=lexicon.get_msd_object())
parser.load()
