import os
from typing import List, Tuple
import torch
from transformers import BertTokenizerFast, BertModel
from . import _device, logger


dumitrescu_bert_v1 = 'dumitrescustefan/bert-base-romanian-cased-v1'
zero_word = '_ZERO_'
unk_word = '_UNK_'
start_word = '_START_'
end_word = '_END_'


class RoBERTModel(object):
    def __init__(self, path_or_name: str, fine_tune: bool = False):
        self._ro_bert_tokenizer: BertTokenizerFast = \
            BertTokenizerFast.from_pretrained(path_or_name)
        self._ro_bert_model: BertModel = BertModel.from_pretrained(path_or_name)
        self._ro_bert_model.to(_device)

        if os.path.isdir(path_or_name):
            # Loading from saved folder, for runtime use.
            self._ro_bert_model.eval()
            self._fine_tune = False
        elif not fine_tune:
            # Do not touch the underlying BERT model,
            # loaded from HuggingFace
            for param in self._ro_bert_model.parameters():
                param.requires_grad = False
            # end for

            self._ro_bert_model.eval()
            self._fine_tune = False
        else:
            # Fine-tune the underlying BERT model
            # For POS tagger and dependency parser
            self._ro_bert_model.train()
            self._fine_tune = True
        # end if

    @property
    def bert_model(self) -> BertModel:
        return self._ro_bert_model

    def _tokenized_bert_embeddings(self, tokens: List[str]) -> torch.Tensor:
        encoding = self._ro_bert_tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        word_ids = encoding.word_ids()
        encoding = {k: v.to(_device) for k, v in encoding.items()}

        if self._fine_tune:
            outputs = self._ro_bert_model(**encoding,
                                          output_hidden_states=True)
        else:
            with torch.inference_mode():
                outputs = self._ro_bert_model(**encoding,
                                              output_hidden_states=True)
            # end with
        # end if

        hidden_states = outputs.hidden_states[-1][0]  # (seq_len, hidden_size)
        
        token_embeddings = []
        current_word_id = None
        current_vectors = []

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            # end if

            if word_id != current_word_id:
                if current_vectors:
                    token_embeddings.append(
                        torch.stack(current_vectors).mean(dim=0))
                # end if

                current_vectors = [hidden_states[idx]]
                current_word_id = word_id
            else:
                current_vectors.append(hidden_states[idx])
            # end if

        if current_vectors:
            token_embeddings.append(torch.stack(current_vectors).mean(dim=0))
        # end if

        return torch.stack(token_embeddings)

    def bert_embeddings(self, tokens: List[Tuple[str, str]]) -> List[torch.Tensor]:
        """Takes a tokenized sentence by Rodna, using RoTokenizer.tokenize(), and returns
        a list of feature vectors computed with the BERT model."""

        result = []
        # Make room for subtokens
        token_step = self._ro_bert_model.config.max_position_embeddings // 2
        
        for i in range(0, len(tokens), token_step):
            if i + token_step <= len(tokens):
                chunk = tokens[i: i + token_step]
            else:
                chunk = tokens[i:]
            # end if

            bert_words = [x[0] for x in chunk
                          if x[1] not in ['SPACE', 'EOL', 'JUNK'] and x[0] not in ['', ' ', '\t']]
            embeddings = self._tokenized_bert_embeddings(tokens=bert_words)
            
            for j in range(len(chunk)):
                word = chunk[j][0]
                tlabel = chunk[j][1]

                if word in ['', ' ', '\t']:
                    bert_features = torch.tensor(self._get_space_vector(),
                                                 dtype=torch.float32).to(_device)
                elif word == '\n' or tlabel == 'EOL':
                    bert_features = torch.tensor(self._get_newline_vector(),
                                                 dtype=torch.float32).to(_device)
                elif tlabel == 'JUNK':
                    bert_features = torch.tensor(self._get_unk_word_vector(),
                                                 dtype=torch.float32).to(_device)
                elif word == zero_word:
                    bert_features = torch.tensor(self._get_zero_word_vector(),
                                                 dtype=torch.float32).to(_device)
                elif word == unk_word:
                    bert_features = torch.tensor(self._get_unk_word_vector(),
                                                 dtype=torch.float32).to(_device)
                elif word == start_word:
                    bert_features = torch.tensor(self._get_start_word_vector(),
                                                 dtype=torch.float32).to(_device)
                elif word == end_word:
                    bert_features = torch.tensor(self._get_end_word_vector(),
                                                 dtype=torch.float32).to(_device)
                else:
                    if bert_words and word == bert_words[0]:
                        bert_features = embeddings[0]
                        bert_words.pop(0)
                        embeddings = embeddings[1:]
                    else:
                        logger.error(
                            f'Out of sync at index [{j}] with word [{word}] and BERT word [{bert_words[0]}]')
                        bert_features = torch.tensor(self._get_unk_word_vector(),
                                                     dtype=torch.float32).to(_device)
                    # end if
                # end if

                result.append(bert_features)
            # end for
        # end for

        return result

    def save(self, destination_folder: str):
        self._ro_bert_tokenizer.save_pretrained(
            save_directory=destination_folder)
        self._ro_bert_model.save_pretrained(
            save_directory=destination_folder)

    def get_embedding_size(self) -> int:
        return self._ro_bert_model.config.hidden_size

    def _get_zero_word_vector(self) -> List[float]:
        return [0.] * self._ro_bert_model.config.hidden_size

    def _get_unk_word_vector(self) -> List[float]:
        return [0.5] * self._ro_bert_model.config.hidden_size

    def _get_start_word_vector(self) -> List[float]:
        """The vector for the start sentence anchor."""
        return [1.0] * self._ro_bert_model.config.hidden_size

    def _get_end_word_vector(self) -> List[float]:
        """The vector for the end sentence anchor."""
        return [1.0] * self._ro_bert_model.config.hidden_size

    def _get_space_vector(self) -> List[float]:
        """The vector for the end sentence anchor."""
        return [0.1] * self._ro_bert_model.config.hidden_size

    def _get_newline_vector(self) -> List[float]:
        """The vector for the end sentence anchor."""
        return [0.7] * self._ro_bert_model.config.hidden_size
