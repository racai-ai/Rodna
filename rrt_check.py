# Checks UD_Romanian-RRT for different types of errors
import sys
import re
from typing import List, Tuple, Dict, Set
from rodna import logger
from rodna.processor.lexicon import Lex
from rodna.processor.tokenizer import RoTokenizer
from rodna.utils.datafile import read_conllu_file

sys.stderr.reconfigure(encoding='utf-8')
lexicon = Lex()
tokenizer = RoTokenizer(lexicon=lexicon)


def check_word_in_lexicon(conllu: List[Tuple[List[str], List[List[str]], List[int]]]):
    lexicon_problems: Dict[str, List[int]] = {}
    lexicon_misses: Dict[str, List[int]] = {}
    lexicon_add: Dict[Tuple[str, str, str], List[int]] = {}

    for _, sentence, line_numbers in conllu:
        for i in range(len(sentence)):
            word = sentence[i][1]
            lemma = sentence[i][2]
            msd = sentence[i][4]
            deprel = sentence[i][7]
            line_no = line_numbers[i]

            if deprel != 'punct' and not lexicon.msd_object.is_valid_msd(msd=msd):
                raise RuntimeError(f'MSD [{msd}] is invalid at line number [{line_no}]')
            # end if

            if deprel != 'punct' and \
                    not (
                        # Skip punctuation
                        re.search('\\d', word) or
                        # Single letter abbreviations, e.g. A.
                        re.fullmatch('[A-Z]\\.', word) or
                        # Proper nouns or foreign stuff
                        msd in ['Np', 'X', 'I', 'Nc']
                    ):
                if not lexicon.is_lex_word(word=word, dash_match=True):
                    if re.match('^[NA]|^Vm|^Rg', msd):
                        wlmk = ('+\t' + word, lemma, msd)

                        if wlmk not in lexicon_add:
                            lexicon_add[wlmk] = []
                        # end if

                        lexicon_add[wlmk].append(line_no)
                    else:
                        lm = f'Word [{word}] with MSD [{msd}] is not in the lexicon'

                        if lm not in lexicon_misses:
                            lexicon_misses[lm] = [line_no]
                        else:
                            lexicon_misses[lm].append(line_no)
                        # end if
                    # end if

                    continue
                # end if

                word_a_class = lexicon.get_word_ambiguity_class(word=word, exact_match=False, dash_match=True)

                if msd not in word_a_class:
                    if re.match('^[NA]|^Vm|^Rg', msd):
                        wlmk = ('?\t' + word, lemma, msd)

                        if wlmk not in lexicon_add:
                            lexicon_add[wlmk] = []
                        # end if

                        lexicon_add[wlmk].append(line_no)
                    else:
                        lm = f'MSD [{msd}] does not belong to word [{word}] ambiguity class'

                        if lm not in lexicon_problems:
                            lexicon_problems[lm] = [line_no]
                        else:
                            lexicon_problems[lm].append(line_no)
                        # end if
                    # end if

                    continue
                # end if

                word_lemmas = lexicon.get_word_lemma(word=word, msd=msd)

                if lemma not in word_lemmas:
                    lm = f'Lemma [{lemma}] is not in lexicon for word [{word}] and MSD [{msd}]'

                    if lm not in lexicon_problems:
                        lexicon_problems[lm] = [line_no]
                    else:
                        lexicon_problems[lm].append(line_no)
                    # end if
                # end if
            # end if
        # end for
    # end for

    sorted_lexicon_problems = sorted(
        lexicon_problems.keys(),
        key=lambda x: len(lexicon_problems[x]), reverse=True)
    
    logger.info("\n\nLexicon problems:\n")

    for lm in sorted_lexicon_problems:
        logger.error(f'{len(lexicon_problems[lm])} time(s) -- ' +
                     lm + f', at line numbers {lexicon_problems[lm]}')
    # end for

    logger.info("\n\nNot in lexicon:\n")

    sorted_lexicon_misses = sorted(
        lexicon_misses.keys(),
        key=lambda x: len(lexicon_misses[x]), reverse=True)
    
    for lm in sorted_lexicon_misses:
        logger.warning(f'{len(lexicon_misses[lm])} time(s) -- ' +
                       lm + f', at line numbers {lexicon_misses[lm]}')
    # end for

    logger.info("\n\nAdd to lexicon:\n")

    for wlm in lexicon_add:
        print('\t'.join(wlm) + '\t' + str(lexicon_add[wlm]), file=sys.stderr)
    # end for


def check_dash_tokenization(conllu: List[Tuple[List[str], List[List[str]], List[int]]]):
    current_split: List[str] = []

    for _, sentence, line_numbers in conllu:
        for i in range(len(sentence)):
            word = sentence[i][1]
            line_no = line_numbers[i]

            if word != '-':
                if  word.startswith('-') or word.endswith('-'):
                    if not current_split:
                        current_split.append(word)
                    else:
                        if word.startswith('-') and current_split[0].startswith('-'):
                            current_split.append(word)
                        elif word.endswith('-') and current_split[0].endswith('-'):
                            current_split.append(word)
                        else:
                            # -a + b- encountered
                            if len(current_split) == 1:
                                current_split.insert(0, sentence[i - 2][1])
                            elif len(current_split) == 2:
                                current_split.insert(0, sentence[i - 3][1])
                            # end if

                            dash_word = ''.join(current_split)
                            rotk_current_split = tokenizer._decide_dash_split(
                                word=dash_word)
                            rotk_current_split = [x[0] for x in rotk_current_split]

                            if rotk_current_split != current_split:
                                logger.error(
                                    f'Tokenization error {current_split} at line [{line_no}]: RODNA produces {rotk_current_split}')
                            # end if

                            current_split.clear()
                            current_split.append(word)
                        # end if
                    # end if
                elif current_split and '-' in current_split[0]:
                    if current_split[0].endswith('-'):
                        current_split.append(word)
                    elif len(current_split) == 1:
                        current_split.insert(0, sentence[i - 2][1])
                    elif len(current_split) == 2:
                        current_split.insert(0, sentence[i - 3][1])
                    # end if 

                    dash_word = ''.join(current_split)
                    rotk_current_split = tokenizer._decide_dash_split(word=dash_word)
                    rotk_current_split = [x[0] for x in rotk_current_split]
                    
                    if rotk_current_split != current_split:
                        logger.error(
                            f'Tokenization error {current_split} at line [{line_no}]: RODNA produces {rotk_current_split}')
                    # end if

                    current_split.clear()
                # end if
            # end if
        # end for
    # end for

    a = 1


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python rrt_check.py <.conllu file from RRT>')
        sys.exit(1)
    # end if

    rrt_split = read_conllu_file(input_file=sys.argv[1])
    check_dash_tokenization(conllu=rrt_split)
    check_word_in_lexicon(conllu=rrt_split)
