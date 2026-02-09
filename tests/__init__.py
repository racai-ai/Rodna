from processor.tokenizer import RoTokenizer
from processor.splitter import RoSentenceSplitter
from processor.morphology import RoInflect
from processor.lemmatization import RoLemmatizer
from processor.tagger import RoPOSTagger
from processor.parser import RoDepParser
from processor.lexicon import Lex

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
