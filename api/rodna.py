"""This is the main entry into the Rodna module.
It processes UTF-8 text files and outputs CoNLL-U files."""

import sys
from typing import List, Tuple
from conllu.models import Token, TokenList, SentenceList
from rodna.tokenizer import RoTokenizer
from rodna.splitter import RoSentenceSplitter
from rodna.morphology import RoInflect
from rodna.lemmatization import RoLemmatizer
from rodna.tagger import RoPOSTagger
from rodna.parser import RoDepParser
from utils.Lex import Lex
from . import ConlluProcessor

class RodnaProcessor(ConlluProcessor):
    """Instantiates all parts needed to process a Romanian text."""

    def __init__(self, with_punct_ctags: bool = True) -> None:
        self._use_punct_ctags = with_punct_ctags
        self.lexicon = Lex()
        self.tokenizer = RoTokenizer(self.lexicon)
        self.splitter = RoSentenceSplitter(self.lexicon, self.tokenizer)
        self.splitter.load()
        self.morphology = RoInflect(self.lexicon)
        self.morphology.load()
        self.tagger = RoPOSTagger(
            self.lexicon, self.tokenizer, self.morphology, self.splitter)
        self.tagger.load()
        self.lemmatizer = RoLemmatizer(self.lexicon, self.morphology)
        self.parser = RoDepParser(msd=self.lexicon.get_msd_object())
        self.parser.load()

    def _check_input_sentence(self, sentence: List[Tuple]) -> bool:
        """If it is an empty sentence (only spaces), skip it."""

        for tok, ttag in sentence:
            if ttag != 'SPACE' and ttag != 'EOL':
                return True
            # end if
        # end for

        return False

    def process_text(self, text: str) -> SentenceList:
        """This is Rodna's text processing method."""

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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: python -m api.rodna <input .txt file>',
              file=sys.stderr, flush=True)
        exit(1)
    # end if

    input_file = sys.argv[1]
    rodna = RodnaProcessor()
    rodna.process_text_file(txt_file=input_file)
