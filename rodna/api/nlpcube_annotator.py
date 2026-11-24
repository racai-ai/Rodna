"""Uses NLP-Cube annotator to process Romanian text to CoNLL-U files.
`pip install nlpcube` first!"""

import re
import os
import sys
from typing import Tuple
from pathlib import Path
from cube.api import CubeObj, _download_model
from conllu.models import Token, TokenList, SentenceList
from typing import override
from . import ConlluProcessor, space_rx



class NLPCubeProcessor(ConlluProcessor):
    def __init__(self):
        cube_cache_dir = os.path.join('data', 'other', 'nlpcube')

        if not os.path.isdir(cube_cache_dir):
            cube_cache_dir = os.path.join(
                str(Path.home()), '.nlpcube', '3.0', 'ro')
            
            if not os.path.isdir(cube_cache_dir):
                os.makedirs(name=cube_cache_dir)
            # end if

            _download_model(cube_cache_dir, 'ro')
        # end if

        cube_model_prefix = os.path.join(cube_cache_dir, 'ro')
        self._nlp = CubeObj(model_base=cube_model_prefix, device='cuda', lang='ro')

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
                t_sentence = re.sub(f'({prev_token_esc})\\s+({token_esc})', repl='\\1\\2', string=t_sentence)
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
        document = self._nlp(text=text)
        conllu_sentences = SentenceList()
        sid = 1

        for sentence in document.sentences:
            conllu_text = sentence.text
            conllu_text = space_rx.sub(' ', conllu_text)
            sent_id = str(sid)
            conllu_sentence = TokenList()

            for word in sentence.words:
                tdict = {}
                tdict['id'] = word.index
                tdict['form'] = word.word.replace('\t', ' ')
                tdict['lemma'] = word.lemma
                tdict['upos'] = word.upos
                tdict['xpos'] = word.xpos
                tdict['feats'] = word.attrs
                tdict['head'] = word.head.item()
                tdict['deprel'] = word.label
                tdict['deps'] = '_'
                tdict['misc'] = '_'

                # Fix NLP-Cube lemmatizer bug where punctuation
                # is getting random lemmas, consisting of other punctuation chars...
                if tdict['upos'] == 'PUNCT':
                    tdict['lemma'] = tdict['form']
                # end if

                conllu_sentence.append(Token(tdict))
            # end all tokens

            fixed_conllu_sentence, fixed_conllu_text = \
                self._fix_spaces(c_sentence=conllu_sentence, t_sentence=conllu_text)

            conllu_sentence.metadata = {
                'sent_id': sent_id,
                'text': fixed_conllu_text
            }

            conllu_sentences.append(fixed_conllu_sentence)
            sid += 1
        # end for sentences

        return conllu_sentences

    @override
    def _correct_sentences(self, sentences: SentenceList) -> SentenceList:
        # TODO: implement post-processing steps
        return sentences


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m rodna.api.nlpcube_annotator <input .txt file>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if

    input_file = sys.argv[1]
    cubep = NLPCubeProcessor()
    cubep.process_text_file(txt_file=input_file)
