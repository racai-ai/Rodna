from typing import List, Tuple, Set
import os
import sys
from tqdm import tqdm
from .tokenizer import RoTokenizer
from utils.Lex import Lex
from .parserone import RoDepParserTree
from .parsertwo import RoDepParserLabel
from utils.MSD import MSD
from config import PARSER_DEPRELS_FILE, PARSER_MODEL_FOLDER
from . import logger


def read_parsed_file(file: str) -> List[List[Tuple[str, str, int, str]]]:
    """Will read in file and return a sequence of tokens from it
    each token with its assigned MSD and dependency information."""

    logger.info(f"Reading training file [{file}]")

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
                logger.warning(f"Line [{line_count}] in file [{file}] is not well-formed!")
            else:
                current_sentence.append(
                    (parts[1], parts[3], int(parts[4]), parts[5]))
            # end if
        # end all lines
    # end with

    return sentences


class RoDepParser(object):

    def __init__(self, msd: MSD, tok: RoTokenizer):
        self._tokenizer = tok
        self._rodep1 = RoDepParserTree(msd, tok)
        self._deprels = self._load_deprels()
        self._rodep2 = RoDepParserLabel(msd, tok, self._deprels)

    def parse_sentence(self, sentence: List[Tuple]) -> List[Tuple]:
        """This is the main entry into the Romanian depencency parser.
        Takes a POS tagged sentence (tokens are tuples of word, MSD, prob)
        and returns its parsed version."""

        # Corner case: sentence with 1 word
        if len(sentence) == 1:
            return [(sentence[0][0], sentence[0][1], 0, 'root', {'LINK': 1., 'DREL': 1.})]
        # end if

        tree_sentence = self._rodep1.parse_sentence(sentence)

        # Unlabeled parsing returns the link probability as the last element of tuple
        # Remove it for the call to find_sentence_paths()
        tree_sentence_no_probs = [
            (word, msd, head, drel) for word, msd, head, drel, tree_prob in tree_sentence]
        tree_paths = self._rodep2.find_sentence_paths(
            sentence=tree_sentence_no_probs)
        parsed_sentence = [(word, msd, head, {}, tree_prob)
                           for word, msd, head, _, tree_prob in tree_sentence]

        for path in tree_paths:
            path_deprels = self._rodep2.label_path(
                tree_sentence_no_probs, path)
            
            for i, (dr, dp) in enumerate(path_deprels):
                si = path[i]
                si_drels = parsed_sentence[si][3]

                if dr not in si_drels:
                    si_drels[dr] = [dp, 1]
                else:
                    si_drels[dr][0] += dp
                    si_drels[dr][1] += 1
                # end if
            # end for
        # end all paths

        # This is the final parse tree, along with dependency labels
        fully_parsed_sentence = []

        for word, msd, head, deprels, tr_prob in parsed_sentence:
            best_dr = ''
            best_dr_prob = 0.

            for dr in deprels:
                dr_prob = deprels[dr][0] / deprels[dr][1]

                if dr_prob > best_dr_prob:
                    best_dr_prob = dr_prob
                    best_dr = dr
                # end if
            # end for

            fully_parsed_sentence.append((word, msd, head, best_dr, {'LINK': tr_prob, 'DREL': dr_prob}))
        # end for

        return fully_parsed_sentence

    def equal_deprels(self, pars_sent: List[Tuple], gold_sent: List[Tuple], index: int) -> bool:
        wp, mp, hp, drp, scp = pars_sent[index]
        wg, mg, hg, drg = gold_sent[index]

        if drp == drg:
            return True
        # end if

        drp_super = ''
        drg_super = ''

        if ':' in drp:
            # 1. Exact match
            drp_super, _ = drp.split(':')
        # end if

        if ':' in drg:
            drg_super, _ = drg.split(':')
        # end if

        if (drp_super and drp_super == drg) or \
                (drg_super and drg_super == drp) or \
                (drp_super and drg_super and drp_super == drg_super):
            # 2. General UD relations match, e.g. obl == obl:pmod
            return True
        # end if

        # 3. Other RRT-based, inconsistent annotations
        if drp == 'amod' and (mp.startswith('Afp') or mp.startswith('Ya')) and \
                (drg == 'fixed' or drg == 'flat'):
            return True
        # end if

        if ((drp == 'amod' and drg == 'acl') or \
                (drp == 'acl' and drg == 'amod')) and \
                mp.startswith('Vmp'):
            return True
        # end if

        return False

    def do_uas_and_las_eval(self, sentences: List[List[Tuple]], ml_type: str, relaxed: bool = False):
        """Does the Unlabeled/Labeled Attachment Score calculation, given a dev/test set.
        If `relaxed is True`, some more relaxed approach is taken for dependency labels comparison.
        See the equal_deprels() method for details."""

        correct_uas = 0
        correct_las = 0
        all_links = 0
        debug_file = os.path.join(PARSER_MODEL_FOLDER, f'parser-debug-{ml_type}.txt')

        with open(debug_file, mode='w', encoding='utf-8') as f:
            for gold_snt in tqdm(sentences, desc=f'UAS/LAS on [{ml_type}] set'):
                tag_snt = [(word, msd, 1.0)
                        for (word, msd, head, deprel) in gold_snt]
                par_snt = self.parse_sentence(sentence=tag_snt)

                for i in range(len(par_snt)):
                    if par_snt[i][2] == gold_snt[i][2]:
                        # That is, the head is the same
                        correct_uas += 1

                        if relaxed:
                            if self.equal_deprels(par_snt, gold_snt, index=i):
                                correct_las += 1
                            # end if
                        elif par_snt[i][3] == gold_snt[i][3]:
                            correct_las += 1
                        # end if
                    # end if

                    print(f'{i + 1}\t{gold_snt[i][0]}\t{gold_snt[i][1]}', file=f, end='')
                    do_we_have_a_problem = False

                    if par_snt[i][2] == gold_snt[i][2]:
                        print(f'\t{gold_snt[i][2]}', file=f, end='')
                    else:
                        print(
                            f'\t<{gold_snt[i][2]},{par_snt[i][2]}>', file=f, end='')
                        do_we_have_a_problem = True
                    # end if

                    if par_snt[i][3] == gold_snt[i][3]:
                        print(f'\t{gold_snt[i][3]}', file=f, flush=True, end='')
                    else:
                        print(
                            f'\t[{gold_snt[i][3]},{par_snt[i][3]}]', file=f, end='')
                        do_we_have_a_problem = True
                    # end if

                    if do_we_have_a_problem:
                        link_prob = par_snt[i][4]['LINK']
                        drel_prob = par_snt[i][4]['DREL']
                        
                        print(
                            f'\tLINK={link_prob:.5f},DREL={drel_prob:.5f}', file=f, flush=True)
                    else:
                        print(file=f, flush=True)
                    # end if
                # end for

                print(file=f, flush=True)
                all_links += len(gold_snt)
            # end for
        # end with

        uas = correct_uas / all_links
        las = correct_las / all_links

        logger.info(f'UAS = [{uas:.5f}], LAS = [{las:.5f}] on [{ml_type}] set')

    def train(self,
            train_sentences: List[List[Tuple]],
            dev_sentences: List[List[Tuple]], test_sentences: List[List[Tuple]]):
        """Performs RoDepParserTree and RoDepParserLabel training."""

        self._rodep1.train(train_sentences, dev_sentences, test_sentences)
        self._rodep2.train(train_sentences, dev_sentences, test_sentences)

    def load(self):
        self._rodep1.load()
        self._rodep2.load()

    def _load_deprels(self) -> Set[str]:
        deprels = set()

        with open(PARSER_DEPRELS_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                deprels.add(line.strip())
            # end for
        # end with

        return deprels


if __name__ == '__main__':
    lex = Lex()
    tok = RoTokenizer(lex)
    msd = MSD()
    par = RoDepParser(msd=msd, tok=tok)

    # For a given split, like in RRT
    training_file = os.path.join(
        "data", "training", "parser", "ro_rrt-ud-train.tab")
    training = read_parsed_file(file=training_file)

    development_file = os.path.join(
        "data", "training", "parser", "ro_rrt-ud-dev.tab")
    development = read_parsed_file(file=development_file)

    testing_file = os.path.join(
        "data", "training", "parser", "ro_rrt-ud-test.tab")
    testing = read_parsed_file(file=testing_file)

    par.train(train_sentences=training, dev_sentences=development, test_sentences=testing)

    # Performance
    par.load()
    par.do_uas_and_las_eval(sentences=development, ml_type='dev')
    par.do_uas_and_las_eval(sentences=testing, ml_type='test')
