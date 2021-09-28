import os
from utils.datafile import read_conllu_file, conllu_corpus_to_tab_file

# Just input the RRT .conllu files here, depending on where they are for you.
# Make sure you run this script every time UD_Romanian-RRT changes!
rrt_train_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt-ud-train.conllu")
rrt_dev_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt-ud-dev.conllu")
rrt_test_file = os.path.join(
    "..", "UD", "UD_Romanian-RRT", "ro_rrt-ud-test.conllu")
rrt_train_tab = os.path.join(
    "data", "training", "tagger", "ro_rrt-ud-train.tab")
rrt_dev_tab = os.path.join(
    "data", "training", "tagger", "ro_rrt-ud-dev.tab")
rrt_test_tab = os.path.join(
    "data", "training", "tagger", "ro_rrt-ud-test.tab")

corpus = read_conllu_file(rrt_train_file)
conllu_corpus_to_tab_file(corpus, rrt_train_tab)
corpus = read_conllu_file(rrt_dev_file)
conllu_corpus_to_tab_file(corpus, rrt_dev_tab)
corpus = read_conllu_file(rrt_test_file)
conllu_corpus_to_tab_file(corpus, rrt_test_tab)
