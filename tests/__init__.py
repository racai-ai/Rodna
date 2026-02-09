from rodna.tokenizer import RoTokenizer
from rodna.splitter import RoSentenceSplitter
from rodna.morphology import RoInflect
from rodna.lemmatization import RoLemmatizer
from rodna.tagger import RoPOSTagger
from rodna.parser import RoDepParser
from rodna.lexicon import Lex

lexicon = Lex()
tokenizer = RoTokenizer(lexicon=lexicon)
splitter = RoSentenceSplitter(lexicon=lexicon, tokenizer=tokenizer)
splitter.load()
morphology = RoInflect(lexicon=lexicon)
morphology.load()
tagger = RoPOSTagger(lexicon=lexicon, tokenizer=tokenizer,
                     morphology=morphology, splitter=splitter)
tagger.load()
lemmatizer = RoLemmatizer(lexicon=lexicon, inflector=morphology)
parser = RoDepParser(msd_desc=lexicon.get_msd_object(), tokenizer=tokenizer)
parser.load()
