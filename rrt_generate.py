import os
from utils.datafile import read_conllu_file, conllu_corpus_to_tab_file
from rodna.tokenizer import RoTokenizer
from utils.Lex import Lex

# In the ro_rrt-ud-train2.conllu file all sentence final
# tokens have been replaced with <token>#SE#, like so:
# 1	Știa	ști	VERB	Vmii3s	Mood=Ind|Number=Sing|Person=3|Tense=Imp|VerbForm=Fin	0	root	_	_
# 2	și	și	CCONJ	Crssp	Polarity = Pos	4	advmod	_	_
# 3	de	de	ADP	Spsa	AdpType = Prep | Case = Acc	4	case	_	_
# 4	ce	ce	PRON	Pw3--r	Case = Acc, Nom | Person = 3 | PronType = Int, Rel	1	ccomp	_	SpaceAfter = No
# 5	.#SE#	.	PUNCT	PERIOD	_	1	punct	_	_
# Then UD/tools/conllu_to_text.pl is used to get the text out from this file.
# We generate EOS labels using the automatically generated #SE# annotation.
def generate_ssplit_rrt_training(in_file: str, out_file: str):
    with open(in_file, mode='r', encoding='utf-8') as fi:
        rrt_text = ''.join(fi.readlines())
    # end with

    lex = Lex()
    tokenizer = RoTokenizer(lexicon=lex)
    sentences = rrt_text.split('#SE#')

    with open(out_file, mode='w', encoding='utf-8') as fo:
        for i, sentence in enumerate(sentences):
            # conllu_to_text.pl artificially inserts a space after each sentence.
            # Place SENTEND there.
            sentence = sentence.replace('#SE#', '')
            sentence = sentence.lstrip()

            if not sentence:
                continue
            # end if

            if i + 1 < len(sentences):
                sentence += sentences[i + 1][0]
            # end if

            sentence_tokens = tokenizer.tokenize(input_string=sentence)
            sentence_tokens[-1] = (sentence_tokens[-1][0],
                                   sentence_tokens[-1][1], 'SENTEND')

            for tok in sentence_tokens:
                if tok[1] == 'EOL':
                    if len(tok) == 3:
                        tok = (' ', 'EOL', tok[2])
                    else:
                        tok = (' ', 'EOL')
                    # end if
                # end if

                print('\t'.join(tok), file=fo)
            # end for
        # end for
    # end with


# Just input the RRT .conllu files here, depending on where they are for you.
# Make sure you run this script every time UD_Romanian-RRT changes!
rrt_train_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt-ud-train.conllu")
rrt_dev_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt-ud-dev.conllu")
rrt_test_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt-ud-test.conllu")

rrt_train_tag = os.path.join(
    "data", "training", "tagger", "ro_rrt-ud-train.tab")
rrt_dev_tag = os.path.join(
    "data", "training", "tagger", "ro_rrt-ud-dev.tab")
rrt_test_tag = os.path.join(
    "data", "training", "tagger", "ro_rrt-ud-test.tab")

rrt_train_par = os.path.join(
    "data", "training", "parser", "ro_rrt-ud-train.tab")
rrt_dev_par = os.path.join(
    "data", "training", "parser", "ro_rrt-ud-dev.tab")
rrt_test_par = os.path.join(
    "data", "training", "parser", "ro_rrt-ud-test.tab")

corpus = read_conllu_file(rrt_train_file)
conllu_corpus_to_tab_file(corpus, rrt_train_tag, for_tool='tagger')
conllu_corpus_to_tab_file(corpus, rrt_train_par, for_tool='parser')

corpus = read_conllu_file(rrt_dev_file)
conllu_corpus_to_tab_file(corpus, rrt_dev_tag, for_tool='tagger')
conllu_corpus_to_tab_file(corpus, rrt_dev_par, for_tool='parser')

corpus = read_conllu_file(rrt_test_file)
conllu_corpus_to_tab_file(corpus, rrt_test_tag, for_tool='tagger')
conllu_corpus_to_tab_file(corpus, rrt_test_par, for_tool='parser')

# This file is not standard. You have to regenerate it, if needed.
ssplit_txt_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt_train2.txt")
ssplit_out_file = os.path.join("data", "training", "splitter", "ro_rrt-train.txt.tok")
generate_ssplit_rrt_training(in_file=ssplit_txt_file, out_file=ssplit_out_file)
