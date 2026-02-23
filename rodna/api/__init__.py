import sys
import os
from typing import List, Tuple, override
import multiprocessing as mp
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from conllu.models import Token, TokenList, SentenceList
from ..processor import _device, normalize_text
from ..processor.lexicon import Lex
from ..processor.parser import RoDepParser
from ..processor.tagger import RoPOSTagger
from ..processor.lemmatization import RoLemmatizer
from ..processor.morphology import RoInflect
from ..processor.splitter import RoSentenceSplitter
from ..processor.tokenizer import RoTokenizer
from .. import download_resources, logger


class ConlluProcessor(ABC):
    """Introduces a super class of text processing API,
    to facilitate Rodna vs. competition comparison."""

    def process_text_file(self, txt_file: str):
        """Takes a UTF-8 Romanian text file, processes it by
        calling `process_text()` and outputs the .conllu file in
        the same folder."""

        with open(txt_file, mode='r', encoding='utf-8') as f:
            all_text_lines = ''.join(f.readlines())
        # end with

        sentences = self.process_text(text=all_text_lines)
        txt_file_path = Path(txt_file)
        processor_name = self.__class__.__name__.lower()

        if processor_name.endswith('processor'):
            processor_name = processor_name.replace('processor', '')
        # end if

        if not processor_name:
            processor_name = 'generic'
        # end if

        cnl_file_path = txt_file_path.parent / \
            Path(txt_file_path.stem + '.' + processor_name + '.conllu')

        # Print the CoNLL-U file
        with open(cnl_file_path, mode='w', encoding='utf-8') as f:
            for token_list in sentences:
                print(token_list.serialize(), file=f, end='')
            # end for
        # end with

    @abstractmethod
    def process_text(self, text: str) -> SentenceList:
        """To be implemented in subclasses."""
        pass


class RodnaProcessor(ConlluProcessor):
    """This is the main entry into the Rodna module.
    It processes UTF-8 text files and outputs CoNLL-U files."""

    def __init__(self, with_punct_ctags: bool = True, device: torch.device = _device) -> None:
        # Download resources once, if needed
        download_resources()
        
        logger.info(f'Loading Rodna processor on device [{device}], in process [{mp.current_process().name}]')

        self._device = device
        self._use_punct_ctags = with_punct_ctags
        self.lexicon = Lex()
        self.tokenizer = RoTokenizer(self.lexicon)
        self.splitter = RoSentenceSplitter(
            self.lexicon, self.tokenizer, device=self._device)
        self.splitter.load()
        self.morphology = RoInflect(self.lexicon, device=self._device)
        self.morphology.load()
        self.tagger = RoPOSTagger(
            self.lexicon, self.tokenizer, self.morphology, self.splitter, device=self._device)
        self.tagger.load()
        self.lemmatizer = RoLemmatizer(self.lexicon, self.morphology)
        self.parser = RoDepParser(msd_desc=self.lexicon.get_msd_object(),
                                  tokenizer=self.tokenizer, device=self._device)
        self.parser.load()

    def _check_input_sentence(self, sentence: List[Tuple]) -> bool:
        """If it is an empty sentence (only spaces), skip it."""

        for tok, ttag in sentence:
            if ttag != 'SPACE' and ttag != 'EOL':
                return True
            # end if
        # end for

        return False

    @override
    def process_text(self, text: str) -> SentenceList:
        """This is Rodna's text processing method."""
        # 0. Normalize text
        text = normalize_text(text=text)

        # 1. Sentence splitting does tokenization as well.
        sentences = self.splitter.sentence_split(input_text=text)
        conllu_sentences = SentenceList()
        sent_id = 1

        for input_sentence in sentences:
            if not self._check_input_sentence(input_sentence):
                continue
            # end if

            # 2. POS tagging
            tagged_sentence = self.tagger.tag_sentence(sentence=input_sentence)
            # 3. Lemmatization
            lemmatized_sentence = self.lemmatizer.lemmatize_sentence(
                sentence=tagged_sentence)
            # 4. Dependency parsing
            parsed_sentence = self.parser.parse_sentence(
                sentence=tagged_sentence)

            # 4.1 Make sure we get the same processed tokens back
            assert len(tagged_sentence) == len(lemmatized_sentence)
            assert len(lemmatized_sentence) == len(parsed_sentence)

            # 5. Assemble the CoNLL-U sentence
            tidx = 0
            conllu_sentence = TokenList()
            conllu_text = []

            for i, (tok, ttag) in enumerate(input_sentence):
                if tok == tagged_sentence[tidx][0]:
                    tmsd = tagged_sentence[tidx][1]
                    tlem = lemmatized_sentence[tidx][2]
                    tupos = self.lexicon.get_msd_object().msd_to_upos(msd=tmsd, tok_tag=ttag)
                    tfeats = self.lexicon.get_msd_object().msd_to_morpho_feats(msd=tmsd)
                    thead = parsed_sentence[tidx][2]
                    tdrel = parsed_sentence[tidx][3]
                    tdict = {}
                    tdict['id'] = tidx + 1
                    tdict['form'] = tok
                    tdict['lemma'] = tlem
                    tdict['upos'] = tupos

                    if self._use_punct_ctags and tmsd.startswith('Z'):
                        tdict['xpos'] = self.lexicon.get_msd_object(
                        ).get_punct_ctag(tok)
                    else:
                        tdict['xpos'] = tmsd
                    # end if

                    tdict['feats'] = tfeats
                    tdict['head'] = thead
                    tdict['deprel'] = tdrel
                    tdict['deps'] = '_'
                    tdict['misc'] = '_'

                    conllu_text.append(tok)

                    if i + 1 < len(input_sentence):
                        next_tok = input_sentence[i + 1][0]
                        next_ttag = input_sentence[i + 1][1]

                        if next_ttag != 'SPACE' and next_ttag != 'EOL' and \
                                not self.tokenizer.is_space(next_tok):
                            tdict['misc'] = 'SpaceAfter=No'
                        else:
                            conllu_text.append(' ')
                        # end if
                    # end if

                    conllu_sentence.append(Token(tdict))
                    tidx += 1
                # end if proper token

                if tidx == len(tagged_sentence):
                    # Ignore left-over spaces.
                    break
                # end if
            # end all tokenizer tokens (including spaces)

            # Sanity check
            assert len(tagged_sentence) == len(conllu_sentence)

            conllu_sentence.metadata = {
                'sent_id': sent_id,
                'text': ''.join(conllu_text)
            }
            conllu_sentences.append(conllu_sentence)
            sent_id += 1
        # end all sentences

        return conllu_sentences
# end class


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m rodna.api <input .txt file>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if

    input_path = sys.argv[1]

    if os.path.isfile(input_path):
        rodna = RodnaProcessor()
        rodna.process_text_file(txt_file=input_path)
    else:
        print('Usage: python -m rodna.api <input .txt file>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if
