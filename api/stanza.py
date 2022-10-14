"""Uses Stanford's Stanza to process Romanian text to CoNLL-U files."""

import os
import sys
import stanza
from conllu.models import Token, TokenList, SentenceList
from . import ConlluProcessor

class StanzaProcessor(ConlluProcessor):
    def __init__(self):
        stanza_models_base_folder = os.path.join('data', 'other', 'stanza')
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
                os.path.join(stanza_models_base_folder, 'tokenize', 'ro_rrt_tokenizer.pt'),
            'pos_model_path':
                os.path.join(stanza_models_base_folder, 'pos', 'ro_rrt_tagger.pt'),
            'lemma_model_path':
                os.path.join(stanza_models_base_folder, 'lemma', 'ro_rrt_lemmatizer.pt'),
            'depparse_model_path':
                os.path.join(stanza_models_base_folder, 'depparse', 'ro_rrt_parser.pt')
        }
        self._nlp = stanza.Pipeline(**config)

    def process_text(self, text: str) -> SentenceList:
        sentences = self._nlp(doc=text)
        conllu_sentences = SentenceList()
        sent_id = 1

        # Assemble the CoNLL-U sentence
        for input_sentence in sentences:
            conllu_sentence = TokenList()
            conllu_text = []

            for i, tdict in enumerate(input_sentence):
                tdict['form'] = tdict['text']
                tok = tdict['form']
                tdict['deps'] = '_'
                tdict['misc'] = '_'

                conllu_text.append(tok)

                if i + 1 < len(input_sentence):
                    next_tdict = input_sentence[i + 1]

                    if next_tdict['start_char'] == tdict['end_char']:
                        tdict['misc'] = 'SpaceAfter=No'
                    else:
                        conllu_text.append(' ')
                    # end if
                # end if

                del tdict['start_char']
                del tdict['end_char']
                del tdict['text']

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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: python -m api.stanza <input .txt file>',
              file=sys.stderr, flush=True)
        exit(1)
    # end if

    input_file = sys.argv[1]
    stanzap = StanzaProcessor()
    stanzap.process_text_file(txt_file=input_file)
