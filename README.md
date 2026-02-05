## RODNA
**RO**manian **D**eep **N**eural networks **A**rchitectures (RODNA) is a Python 3/PyTorch project with the declared goal of obtaining better results at Romanian text processing through the use of Romanian-specific features than generic, language-independent ML toolkits.

## Training data
Latest version of the Romanian RRT UD corpus available at [UD_Romanian-RRT](https://github.com/UniversalDependencies/UD_Romanian-RRT.git).

Latest training data is pushed to this repository, but if you want to generate fresh training data, run `python3 rrt_generate.py`. Make sure you read the comments preceding the function `def generate_ssplit_rrt_training(in_file: str, out_file: str):` from `rrt_generate.py` first. Folder `UD_Romanian-RRT` **must** be available at `../UD/UD_Romanian-RRT` relative to the folder containing this file.

## Sentence splitter
A Bi-LSTM over a frozen BERT embedding neural network that does sentence splitting (classifies each token as 'end of sentence' or 'not end of sentence').

Precision on 'end of sentence' label is 99.62% on the test split of RRT.\
Recall on 'end of sentence' label is 99.38% on the test split of RRT.\
F1 on 'end of sentence' label is 99.5% on the test split of RRT.

## Romanian morphology
A LSTM neural network than learns the mapping from a word form to its possible [MSDs](https://nl.ijs.si/ME/V6/msd/html/msd-ro.html). It works on character embeddings of the input word, from left to right.

Precision on MSDs that are in the word's ambiguity class is 95.14%.\
Recall of MSDs that are in the word's ambiguity class is 92.66%.\
F1 of the above is 93.88%.

## POS Tagger
A Bi-LSTM-CRF head over a BERT embedding to get coarse-grained POS tags coupled with a Bi-LSTM head over another BERT embedding to get the [MSD](https://nl.ijs.si/ME/V6/msd/html/msd-ro.html) of the current word, given its coarse-grained POS tag. The POS tagger uses Romanian-specific features, extracted beforehand from the input sentence.

### With coarse-grained to fine-grained mapping (called "tiered tagging")
Accuracy on fine-grained POS tags (MSDs) of the dev set is 98.15%.\
Accuracy on fine-grained POS tags (MSDs) of the test set is 97.75%.

### Without tiered tagging (roughly 10 times faster)
Accuracy on fine-grained POS tags (MSDs) of the dev set is 98.05%.\
Accuracy on fine-grained POS tags (MSDs) of the test set is 97.59%.
