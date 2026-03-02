"""Uses Stanford's Stanza to process Romanian text to CoNLL-U files.
`pip install stanza` first!"""

import os
import sys
import stanza
from conllu.models import Token, TokenList, SentenceList
from typing import override
from . import ConlluProcessor


class StanzaProcessor(ConlluProcessor):
    def __init__(self):
        stanza_models_base_folder = os.path.join('data', 'other', 'stanza')

        if os.path.isdir(stanza_models_base_folder):
            config = {
                # Comma-separated list of processors to use
                'processors': 'tokenize,pos,lemma,depparse',
                # Language code for the language to build the Pipeline in
                'lang': 'ro',
                # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
                # You only need model paths if you have a specific model outside of stanza_resources
                # All Romanian models have to be trained and available in data/other/stanza
                # See the paths below.
                'tokenize_model_path':
                    os.path.join(stanza_models_base_folder,
                                 'tokenize', 'ro_rrt_nocharlm_tokenizer.pt'),
                'pos_model_path':
                    os.path.join(stanza_models_base_folder, 'pos',
                                 'ro_rrt_transformer_tagger.pt'),
                'lemma_model_path':
                    os.path.join(stanza_models_base_folder, 'lemma',
                                 'ro_rrt_nocharlm_lemmatizer.pt'),
                'depparse_model_path':
                    os.path.join(stanza_models_base_folder,
                                 'depparse', 'ro_rrt_transformer_parser.pt')
            }
            self._nlp = stanza.Pipeline(**config)
        else:
            self._nlp = stanza.Pipeline(
                lang='ro', processors='tokenize,pos,lemma,depparse')
        # end if

    @override
    def process_text(self, text: str, show_progress: bool = False) -> SentenceList:
        document = self._nlp(doc=text)
        conllu_sentences = SentenceList()
        sent_id = 1

        # Assemble the CoNLL-U sentence
        for input_sentence in document.sentences:
            conllu_sentence = TokenList()
            conllu_text = []

            for i, token in enumerate(input_sentence.tokens):
                # Assume no MWEs here
                word = token.words[0]
                tdict = {}
                tdict['id'] = word.id
                tdict['form'] = word.text
                tdict['lemma'] = word.lemma
                tdict['upos'] = word.upos
                tdict['xpos'] = word.xpos
                tdict['feats'] = word.feats
                tdict['head'] = word.head
                tdict['deprel'] = word.deprel
                tdict['deps'] = '_'
                tdict['misc'] = '_'

                conllu_text.append(word.text)

                if i + 1 < len(input_sentence.tokens):
                    next_token = input_sentence.tokens[i + 1]
                    next_word = next_token.words[0]

                    if next_word.start_char == word.end_char:
                        tdict['misc'] = 'SpaceAfter=No'
                    else:
                        conllu_text.append(' ')
                    # end if
                # end if

                conllu_sentence.append(Token(tdict))
            # end all tokens

            conllu_sentence.metadata = {
                'sent_id': sent_id,
                'text': ''.join(conllu_text)
            }
            conllu_sentences.append(conllu_sentence)
            sent_id += 1
        # end all sentences

        return conllu_sentences

    @override
    def _correct_sentences(self, sentences: SentenceList) -> SentenceList:
        # TODO: implement post-processing steps
        return sentences



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m rodna.api.stanza <input .txt file>', file=sys.stderr, flush=True)
        sys.exit(1)
    # end if

    input_file = sys.argv[1]
    stanzap = StanzaProcessor()
    stanzap.process_text_file(txt_file=input_file)
