from typing import List, Tuple, Set
import os
import sys
from inspect import stack
from tqdm import tqdm
from .parserone import RoDepParserTree
from .parsertwo import RoDepParserLabel
from utils.MSD import MSD


def read_parsed_file(file: str, create_label_set: bool = False) -> Tuple[List[List[Tuple]], Set[str]]:
    """Will read in file and return a sequence of tokens from it
    each token with its assigned MSD and dependency information."""

    print(stack()[
            0][3] + ": reading training file {0!s}".format(file), file=sys.stderr, flush=True)

    deprel_set = set()
    sentences = []
    current_sentence = []
    line_count = 0

    with open(file, mode='r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()

            if not line:
                sentences.append(current_sentence)
                current_sentence = []
                continue
            # end if

            parts = line.split()

            if len(parts) != 6:
                print(stack()[0][3] + ": line {0!s} in file {1!s} is not well-formed!".format(
                    line_count, file), file=sys.stderr, flush=True)
            else:
                current_sentence.append(
                    (parts[1], parts[3], int(parts[4]), parts[5]))
                
                if create_label_set:
                    deprel_set.add(parts[5])
                # end if
            # end if
        # end all lines
    # end with

    return sentences, deprel_set


class RoDepParser(object):

    def __init__(self, msd: MSD, deprels: Set[str]):
        self._rodep1 = RoDepParserTree(msd)
        self._rodep2 = RoDepParserLabel(msd, deprels)

    def parse_sentence(self, sentence: List[Tuple]) -> List[Tuple]:
        pass

    def do_uas_and_las_eval(self, sentences: List[List[Tuple]], desc: str):
        """Does the Unlabeled/Labeled Attachment Score calculation, given a dev/test set."""

        correct_uas = 0
        correct_las = 0
        all_links = 0

        for gold_snt in tqdm(sentences, desc=f'UAS/LAS on {desc}set'):
            tag_snt = [(word, msd, 1.0)
                       for (word, msd, head, deprel) in gold_snt]
            par_snt = self.parse_sentence(sentence=tag_snt)

            for i in range(len(par_snt)):
                if par_snt[i][2] == gold_snt[i][2]:
                    # That is, the head is the same
                    correct_uas += 1

                    if par_snt[i][3] == gold_snt[i][3]:
                        correct_las += 1
                    # end if
                # end if
            # end for

            all_links += len(gold_snt)
        # end for

        uas = correct_uas / all_links
        las = correct_las / all_links

        print(f'UAS = {uas:.5f}, LAS = {las:.5f} on {desc}set',
              file=sys.stderr, flush=True)

    def train(self,
            train_sentences: List[List[Tuple]],
            dev_sentences: List[List[Tuple]], test_sentences: List[List[Tuple]]):
        """Performs parser one and parser two training."""

        self._rodep2.train(train_sentences, dev_sentences, test_sentences)
        #self._rodep1.train(train_sentences, dev_sentences, test_sentences)
        

if __name__ == '__main__':
    # For a given split, like in RRT
    training_file = os.path.join(
        "data", "training", "parser", "ro_rrt-ud-train.tab")
    training, drset = read_parsed_file(file=training_file, create_label_set=True)
    par = RoDepParser(msd=MSD(), deprels=drset)

    development_file = os.path.join(
        "data", "training", "parser", "ro_rrt-ud-dev.tab")
    development, _ = read_parsed_file(file=development_file)

    testing_file = os.path.join(
        "data", "training", "parser", "ro_rrt-ud-test.tab")
    testing, _ = read_parsed_file(file=testing_file)

    par.train(train_sentences=training, dev_sentences=development, test_sentences=testing)
    
    # Debug testing
    #par.load()
    #par.do_uas_and_las_eval(sentences=testing, desc='test')
