import re
import os
from typing import List, Tuple
from rodna.lexicon import MSD
from rodna import logger


def txt_file_to_string(input_file: str) -> str:
    """Takes the input text file and returns a single string out of it."""
    return "".join(txt_file_to_lines(input_file))


def txt_file_to_lines(input_file: str) -> List[str]:
    """Reads in the UTF-8/Latin-2 input_file and returns
    a list of the lines of its contents."""

    file_lines = []
    fp = open(input_file, mode="rb")
    line = fp.readline()

    try:
        line = str(line, encoding="utf-8")

        if line.startswith('\uFEFF'):
            line = line.replace('\uFEFF', '', 1)
        # end if
    except UnicodeDecodeError:
        line = str(line, encoding="latin2")
    # end try

    while line:
        file_lines.append(line)
        line = fp.readline()

        try:
            line = str(line, encoding="utf-8")
        except UnicodeDecodeError:
            line = str(line, encoding="latin2")
        # end try
    # end while

    fp.close()

    return file_lines


def read_all_ext_files_from_dir(input_dir: str, extension: str = '.txt') -> List[str]:
    """Reads all the .txt (by default) UTF-8 files from the input_dir.
    If extension is specified, reads all files with that extension."""

    all_files = os.listdir(input_dir)
    txt_files = filter(lambda x: x[-(len(extension)):] == extension, all_files)
    input_dir_txt_files = []

    for f in txt_files:
        input_dir_txt_files.append(os.path.join(input_dir, f))
    # end for

    return input_dir_txt_files


def tok_file_to_tokens(input_file: str) -> List[Tuple[str, str, str] | Tuple[str, str]]:
    """Will read in a RoTokenizer tokenized file and return a sequence of tokens from it."""

    token_sequence = []
    line_count = 0

    with open(input_file, mode="r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            parts = line.rstrip().split('\t')

            if len(parts) < 2 or len(parts) > 3:
                logger.info(f"Line [{line_count}] in file [{input_file}] is not well-formed!")
            else:
                # In there, EOL tokens are spaces, really.
                # Move them back to proper EOLs.
                if parts[1] == 'EOL':
                    if len(parts) == 3:
                        wtuple = ('\n', 'EOL', parts[2])
                    else:
                        wtuple = ('\n', 'EOL')
                    # end if
                else:
                    wtuple = tuple(parts)
                # end if

                token_sequence.append(wtuple)
            # end if
        # end for
    # end while

    return token_sequence


def write_tok_file(tokens: list, output_file: str) -> None:
    with open(output_file, mode='w', encoding="utf-8") as f:
        for p in tokens:
            if p[1] == "SPACE" or p[1] == "EOL":
                print(" \t{0}".format(p[1]), file=f, flush=True)
            else:
                print("{0}\t{1}".format(p[0], p[1]), file=f, flush=True)
            # end if
        # end all tokens
    # end with


def read_conllu_file(input_file: str) -> List[Tuple[List[str], List[List[str]]]]:
    """Reads a CoNLL-U format file are returns it."""

    logger.info(f"Reading CoNLL-U file [{input_file}]")
    corpus = []

    with open(input_file, mode='r', encoding='utf-8') as f:
        comments = []
        sentence = []
        linecounter = 0

        for line in f:
            linecounter += 1
            line = line.strip()

            if line:
                if not line.startswith('#'):
                    parts = line.split()

                    if len(parts) == 10:
                        sentence.append(parts)
                    else:
                        logger.info(f"CoNLL-U line not well formed at line [{linecounter}] in file [{input_file}]")
                else:
                    comments.append(line)
                # end if
            elif sentence:
                corpus.append((comments, sentence))
                sentence = []
                comments = []
            # end if
        # end for line
    # end with

    return corpus


def conllu_corpus_to_tab_file(corpus: list, output_file: str, for_tool: str) -> None:
    """Converts the CoNLL-U corpus into a .tab file for use with RoPOSTagger.py or RoDepParser.py."""

    punct_rx = re.compile('^\\W+$')

    with open(output_file, mode='w', encoding='utf-8') as f:
        for (_, sent) in corpus:
            for parts in sent:
                wid = int(parts[0])
                word = parts[1]
                lemma = parts[2]
                msd = parts[4]
                head = int(parts[6])
                deprel = parts[7]

                if punct_rx.match(word):
                    if word in MSD.punct_msd_inventory:
                        msd = MSD.punct_msd_inventory[word]
                    else:
                        msd = 'Z'
                    # end if
                # end if

                if for_tool == 'tagger':
                    print(f'{word}\t{lemma}\t{msd}', file=f)
                elif for_tool == 'parser':
                    print(f'{wid}\t{word}\t{lemma}\t{msd}\t{head}\t{deprel}', file=f)
                # end if
            # end for parts

            print('', file=f, flush=True)
        # end for corpus
    # end with
