from rodna.tokenizer import RoTokenizer
from rodna.splitter import RoSentenceSplitter
from rodna.morphology import RoInflect

tokenizer = RoTokenizer()
splitter = RoSentenceSplitter(tokenizer)
splitter.load()
morphology = RoInflect(tokenizer.get_lexicon())
morphology.load()
