"""Uses spaCy annotator to process Romanian text to CoNLL-U files.
`pip install spacy` first!"""

import re
import os
import sys
from typing import Tuple, List
import spacy
from spacy.cli import download
from spacy.util import is_package
from spacy.tokens import Token as SpacyToken
from conllu.models import Token, TokenList, SentenceList
from typing import override
from . import ConlluProcessor, space_rx


spacy.prefer_gpu()


class SpacyProcessor(ConlluProcessor):
    def __init__(self):
        spacy_cache_dir = os.path.join('data', 'other', 'spacy')

        if not os.path.isdir(spacy_cache_dir):
            fallback_model = 'ro_core_news_lg'

            if not is_package(fallback_model):
                download(fallback_model)
            # end if

            self._nlp = spacy.load(fallback_model)
        else:
            self._nlp = spacy.load(spacy_cache_dir)
        # end if

    def _fix_spaces(self, c_sentence: TokenList, t_sentence: str) -> Tuple[TokenList, str]:
        for i in range(len(c_sentence)):
            token = c_sentence[i]
            prev_token = {}
            next_token = {}

            if i > 0:
                prev_token = c_sentence[i - 1]
            # end if

            if i < len(c_sentence) - 1:
                next_token = c_sentence[i + 1]
            # end if

            # mașină , când ...
            if token['form'] in ',.:;?!%)”' and prev_token and \
                    re.search('\\w$', prev_token['form']):
                prev_token['misc'] = 'SpaceAfter=No'
                token_esc = re.escape(token['form'])
                prev_token_esc = re.escape(prev_token['form'])
                t_sentence = re.sub(
                    f'({prev_token_esc})\\s+({token_esc})', repl='\\1\\2', string=t_sentence)
            # end if

            # exemplu ( Gogu sau ...
            if token['form'] in '(„' and next_token and \
                    re.search('^\\w', next_token['form']):
                token['misc'] = 'SpaceAfter=No'
                token_esc = re.escape(token['form'])
                next_token_esc = re.escape(next_token['form'])
                t_sentence = re.sub(
                    f'({token_esc})\\s+({next_token_esc})', repl='\\1\\2', string=t_sentence)
            # end if

            # de- acum
            if token['form'].endswith('-') and next_token and \
                    re.search('^[^\\W\\d_]', next_token['form']):
                token['misc'] = 'SpaceAfter=No'
                token_esc = re.escape(token['form'])
                next_token_esc = re.escape(next_token['form'])
                t_sentence = re.sub(
                    f'({token_esc})\\s+({next_token_esc})', repl='\\1\\2', string=t_sentence)
            # end if

            # să -ți
            if token['form'].startswith('-') and prev_token and \
                    re.search('[^\\W\\d_]$', prev_token['form']):
                prev_token['misc'] = 'SpaceAfter=No'
                token_esc = re.escape(token['form'])
                prev_token_esc = re.escape(prev_token['form'])
                t_sentence = re.sub(
                    f'({prev_token_esc})\\s+({token_esc})', repl='\\1\\2', string=t_sentence)
            # end if
        # end for

        return c_sentence, t_sentence.strip()

    @override
    def process_text(self, text: str, show_progress: bool = False) -> SentenceList:
        # spaCy creates tokens from newlines...
        # making sure this does not happen.
        text = space_rx.sub(' ', text)
        document = self._nlp(text=text)

        if not document.has_annotation("SENT_START"):
            raise RuntimeError(
                "The pipeline did not produce sentence-boundary annotations."
            )
        # end if

        if not document.has_annotation("DEP"):
            raise RuntimeError(
                "The pipeline did not produce dependency annotations."
            )
        # end if

        conllu_sentences = SentenceList()
        
        for sid, sentence in enumerate(document.sents, start=1):
            conllu_text = sentence.text
            conllu_text = space_rx.sub(' ', conllu_text)
            sent_id = str(sid)
            conllu_sentence = TokenList()

            # Map spaCy's document-global token indices to sentence-local CoNLL-U IDs.            
            spacy_tokens: List[SpacyToken] = [token for token in sentence]
            token_id_map = {
                token.i: conllu_id
                for conllu_id, token in enumerate(spacy_tokens, start=1)
            }

            for token in spacy_tokens:
                token_id = token_id_map[token.i]

                lemma = token.lemma_ or "_"
                upos = token.pos_ or "_"
                xpos = token.tag_ or "_"

                morph = str(token.morph)
                feats = morph if morph else "_"

                if token.head == token or token.dep_.lower() == "root":
                    head_id = 0
                elif token.head.i in token_id_map:
                    head_id = token_id_map[token.head.i]
                else:
                    raise ValueError(
                        f"Token {token.text!r} has a head outside its sentence: "
                        f"{token.head.text!r}"
                    )
                # end if

                deprel = token.dep_ or "_"

                if deprel == 'ROOT':
                    deprel = deprel.lower()
                # end if

                misc = "SpaceAfter=No" if not token.whitespace_ else '_'

                tdict = {}
                tdict['id'] = token_id      # ID
                tdict['form'] = token.text  # FORM
                tdict['lemma'] = lemma      # LEMMA
                tdict['upos'] = upos        # UPOS
                tdict['xpos'] = xpos        # XPOS / Romanian MSD
                tdict['feats'] = feats      # FEATS
                tdict['head'] = head_id     # HEAD
                tdict['deprel'] = deprel    # DEPREL
                tdict['deps'] = "_"         # DEPS: enhanced dependencies unavailable
                tdict['misc'] = misc        # MISC

                conllu_sentence.append(Token(tdict))
            # end for tokens

            conllu_sentence.metadata = {
                'sent_id': sent_id,
                'text': conllu_text
            }

            conllu_sentences.append(conllu_sentence)
        # end for sentences

        return conllu_sentences

    @override
    def _correct_sentences(self, sentences: SentenceList) -> SentenceList:
        # TODO: implement post-processing steps
        return sentences


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m rodna.api.spacy_annotator <input .txt file>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if

    input_file = sys.argv[1]
    cubep = SpacyProcessor()
    cubep.process_text_file(txt_file=input_file)
