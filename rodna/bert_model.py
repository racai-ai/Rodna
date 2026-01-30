from typing import List, Tuple
import torch
from transformers import BertTokenizerFast, BertModel
from . import _device, logger

model_name = 'dumitrescustefan/bert-base-romanian-cased-v1'
ro_bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
ro_bert_model = BertModel.from_pretrained(model_name)
ro_bert_model.to(_device)

# Freeze BERT (do not update its parameters)
for param in ro_bert_model.parameters():
    param.requires_grad = False
# end for

zero_word = '_ZERO_'
unk_word = '_UNK_'
start_word = '_START_'
end_word = '_END_'


def _tokenized_bert_embeddings(tokens: List[str]):
    encoding = ro_bert_tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    word_ids = encoding.word_ids()
    encoding = {k: v.to(_device) for k, v in encoding.items()}

    with torch.inference_mode():
        outputs = ro_bert_model(**encoding, output_hidden_states=True)
    # end with

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


def bert_embeddings(tokens: List[Tuple[str, str]]) -> List[torch.Tensor]:
    """Takes a tokenized sentence by Rodna, using RoTokenizer.tokenize(), and returns
    a list of feature vectors computed with the BERT model."""

    result = []
    # Make room for subtokens
    token_step = ro_bert_model.config.max_position_embeddings // 2
    
    for i in range(0, len(tokens), token_step):
        if i + token_step <= len(tokens):
            chunk = tokens[i: i + token_step]
        else:
            chunk = tokens[i:]
        # end if

        bert_words = [x[0] for x in chunk
                      if x[1] not in ['SPACE', 'EOL', 'JUNK'] and x[0] != ' ']
        embeddings = _tokenized_bert_embeddings(tokens=bert_words)
        
        for j in range(len(chunk)):
            word = chunk[j][0]
            tlabel = chunk[j][1]

            if word in ['', ' ', '\t']:
                bert_features = torch.tensor(get_space_vector(),
                                             dtype=torch.float32).to(_device)
            elif word == '\n':
                bert_features = torch.tensor(get_newline_vector(),
                                             dtype=torch.float32).to(_device)
            elif tlabel == 'JUNK':
                bert_features = torch.tensor(get_unk_word_vector(),
                                             dtype=torch.float32).to(_device)
            elif word == zero_word:
                bert_features = torch.tensor(get_zero_word_vector(),
                                             dtype=torch.float32).to(_device)
            elif word == unk_word:
                bert_features = torch.tensor(get_unk_word_vector(),
                                             dtype=torch.float32).to(_device)
            elif word == start_word:
                bert_features = torch.tensor(get_start_word_vector(),
                                             dtype=torch.float32).to(_device)
            elif word == end_word:
                bert_features = torch.tensor(get_end_word_vector(),
                                             dtype=torch.float32).to(_device)
            else:
                if bert_words and word == bert_words[0]:
                    bert_features = embeddings[0]
                    bert_words.pop(0)
                    embeddings = embeddings[1:]
                else:
                    logger.error(
                        f'Out of sync at index [{j}] with word [{word}] and BERT word [{bert_words[0]}]')
                    bert_features = torch.tensor(get_unk_word_vector(),
                                                 dtype=torch.float32).to(_device)
                # end if
            # end if

            result.append(bert_features)
        # end for
    # end for

    return result


def get_embedding_size() -> int:
    return ro_bert_model.config.hidden_size


def get_zero_word_vector() -> List[float]:
    return [0.] * ro_bert_model.config.hidden_size


def get_unk_word_vector() -> List[float]:
    return [0.5] * ro_bert_model.config.hidden_size


def get_start_word_vector() -> List[float]:
    """The vector for the start sentence anchor."""
    return [1.0] * ro_bert_model.config.hidden_size


def get_end_word_vector() -> List[float]:
    """The vector for the end sentence anchor."""
    return [1.0] * ro_bert_model.config.hidden_size


def get_space_vector() -> List[float]:
    """The vector for the end sentence anchor."""
    return [0.1] * ro_bert_model.config.hidden_size


def get_newline_vector() -> List[float]:
    """The vector for the end sentence anchor."""
    return [0.7] * ro_bert_model.config.hidden_size
