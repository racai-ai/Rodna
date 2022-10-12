from typing import List
import os
import re
from utils.datafile import read_conllu_file, conllu_corpus_to_tab_file
from rodna.tokenizer import RoTokenizer
from utils.Lex import Lex

def generate_conllu_ssplit_file(conllu_file: str) -> List[str]:
    sentences = []

    with open(conllu_file, mode='r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# text ') or line.startswith('#text '):
                line = line.strip()
                line = re.sub('^#\\s*text\\s*=\\s*', '', line)
                sentences.append(line)
            # end if
        # end for
    # end with

    lex = Lex()
    tok = RoTokenizer(lexicon=lex)
    output_file = os.path.join('data', 'training', 'splitter', 'rrt_train.txt.tok')

    with open(output_file, mode='w', encoding='utf-8') as f:
        for snt in sentences:
            tokens = tok.tokenize(input_string=snt)

            for token, toktag in tokens:
                print(f'{token}\t{toktag}', file=f)
            # end for
            print(f' \tSPACE\tSENTEND', file=f)
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

generate_conllu_ssplit_file(conllu_file=rrt_train_file)
