"""Uses Trankit annotator to process Romanian text to CoNLL-U files.
`pip install trankit` first!"""

import os
import sys
from pathlib import Path
from trankit import Pipeline
from conllu.models import Token, TokenList, SentenceList
from typing import override
from . import ConlluProcessor, space_rx


class TrankitProcessor(ConlluProcessor):
    def __init__(self):
        trankit_cache_dir = os.path.join('data', 'other', 'trankit')

        if os.path.isdir(trankit_cache_dir):
            trankit_cache_dir = os.path.join('data', 'other', 'trankit')
        else:
            trankit_cache_dir = os.path.join(str(Path.home()), '.trankit')

            if not os.path.isdir(trankit_cache_dir):
                os.makedirs(name=trankit_cache_dir)
            # end if
        # end if

        self._nlp = Pipeline(lang='romanian', gpu=True,
                             cache_dir=trankit_cache_dir)
        
    @override
    def process_text(self, text: str, show_progress: bool = False) -> SentenceList:
        document = self._nlp(input=text)
        conllu_sentences = SentenceList()

        for sentence in document['sentences']:
            conllu_text = sentence['text']
            conllu_text = space_rx.sub(' ', conllu_text)
            sent_id = sentence['id']
            conllu_sentence = TokenList()

            for token in sentence['tokens']:
                tdict = {}
                tdict['id'] = token['id']
                tdict['form'] = token['text'].replace('\t', ' ')
                tdict['lemma'] = token['lemma']
                tdict['upos'] = token['upos']
                tdict['xpos'] = token['xpos']
                
                if 'feats' in token:
                    tdict['feats'] = token['feats']
                else:
                    tdict['feats'] = '_'
                # end if

                tdict['head'] = token['head']
                tdict['deprel'] = token['deprel']
                tdict['deps'] = '_'

                span_right_index = token['span'][-1]
                
                if span_right_index < len(conllu_text):
                    span_last_char = conllu_text[span_right_index]
                else:
                    span_last_char = ' '
                # end if

                if space_rx.fullmatch(span_last_char):
                    tdict['misc'] = '_'
                else:
                    tdict['misc'] = 'SpaceAfter=No'
                # end if

                conllu_sentence.append(Token(tdict))
            # end all tokens

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
        print('Usage: python -m rodna.api.trankit_annotator <input .txt file>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if

    input_file = sys.argv[1]
    trankitp = TrankitProcessor()
    trankitp.process_text_file(txt_file=input_file)
