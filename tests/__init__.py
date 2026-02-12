from rodna.processor.tokenizer import RoTokenizer
from rodna.processor.splitter import RoSentenceSplitter
from rodna.processor.morphology import RoInflect
from rodna.processor.lemmatization import RoLemmatizer
from rodna.processor.tagger import RoPOSTagger
from rodna.processor.parser import RoDepParser
from rodna.processor.lexicon import Lex

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
