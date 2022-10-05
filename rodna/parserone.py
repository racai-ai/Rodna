from typing import List, Tuple, Union
import sys
import os
from inspect import stack
from random import shuffle
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from . import _device
from utils.mst import chu_liu_edmonds
from utils.MSD import MSD
from config import PARSER_MODEL_FOLDER, \
    PARSER1_BERT_MODEL_FOLDER, PARSER1_TOKEN_MODEL_FOLDER

class RoBERTHeadFinder(nn.Module):
    """This class finds the probability distribution of possible heads
    for the current token. Uses a MLM BERT model for word embeddings
    which is fine-tuned to find head information."""

    _conf_lstm_size = 512
    _conf_drop_prob = 0.2

    def __init__(self, msd_size: int, head_window_size: int):
        """`msd_size` - the size of the MSD vector, from MSD.msd_reference_vector()
        `head_window_size`: how much to go left/right from current token to search for its head"""

        super().__init__()

        self._nonlin_fn = nn.LeakyReLU()
        self._drop = nn.Dropout(p=RoBERTHeadFinder._conf_drop_prob)
        self._logsoftmax = nn.LogSoftmax(dim=2)

        self._nn_lstm = nn.LSTM(
            input_size=768, hidden_size=RoBERTHeadFinder._conf_lstm_size,
            num_layers=1, batch_first=True, bidirectional=True)

        linear_input_size_0 = msd_size + 2 * RoBERTHeadFinder._conf_lstm_size
        linear_input_size_1 = linear_input_size_0 // 2

        self._nn_linear_0 = nn.Linear(
            in_features=linear_input_size_0,
            out_features=linear_input_size_1, dtype=torch.float32)
        self._nn_linear_1 = nn.Linear(
            in_features=linear_input_size_1,
            out_features=2 * head_window_size + 1, dtype=torch.float32)

        self.to(_device)

    def forward(self, x):
        x_b, x_m = x
        b_size = x_b.shape[0]
        h_0 = torch.zeros(
            2, b_size,
            RoBERTHeadFinder._conf_lstm_size).to(device=_device)
        c_0 = torch.zeros(
            2, b_size,
            RoBERTHeadFinder._conf_lstm_size).to(device=_device)

        the_output, (h_n, c_n) = self._nn_lstm(x_b, (h_0, c_0))
        the_output = torch.cat([the_output, x_m], dim=2).to(device=_device)
        the_output = self._nn_linear_0(the_output)
        the_output = self._nonlin_fn(the_output)
        the_output = self._drop(the_output)
        the_output = self._nn_linear_1(the_output)
        the_output = self._logsoftmax(the_output)

        return the_output


class DPDataset(Dataset):
    """This is a dependency parser dataset. Sorts the sentences by length
    so that they can form batches for LSTMs/GRUs. Shuffles the lengths so that
    we have a shuffled dataset."""

    def __init__(self, sentences: List[List[Tuple]]):
        super().__init__()
        self._sentences_by_length = {}

        for snt in sentences:
            sntlen = len(snt)

            if sntlen not in self._sentences_by_length:
                self._sentences_by_length[sntlen] = []
            # end if

            self._sentences_by_length[sntlen].append(snt)
        # end for

        self.reshuffle()

    def reshuffle(self):
        sentence_lengths = list(self._sentences_by_length.keys())
        shuffle(sentence_lengths)
        self._sentences = []

        for sntlen in sentence_lengths:
            shuffle(self._sentences_by_length[sntlen])
            self._sentences.extend(self._sentences_by_length[sntlen])
        # end for

    def __len__(self) -> int:
        return len(self._sentences)

    def __getitem__(self, index):
        return self._sentences[index]


class RoDepParserTree(object):
    """This is the RODNA dependency parser, based on BERT head finding coupled with
    MST parsing tree discovery. Only gives the unlabeled parse tree of a sentence."""

    _conf_model_file = 'modelone.pt'
    _conf_bert = 'dumitrescustefan/bert-base-romanian-cased-v1'
    # Take 2 * head window + 1 for the output vector dimension
    _conf_head_window = 70
    # Initial learning rate
    _conf_lr = 5e-5
    # Multiplying factor between epochs of the LR
    _conf_gamma_lr = 0.75
    _conf_epochs = 15

    def __init__(self, msd: MSD) -> None:
        super().__init__()
        self._msd = msd
        self._headmodel = RoBERTHeadFinder(
            msd_size=self._msd.get_output_vector_size(),
            head_window_size=RoDepParserTree._conf_head_window)

    def _process_sentence(self, sentence: List[Tuple], runtime: bool = False) -> Tuple[Union[Tensor, None]]:
        """Gets a sentence and returns a tuple of tensors to be run
        through the neural network, as part of a batch.
        Shape of tuple of inputs is [1, len(sentence), feature_size].
        When `runtime is True`, the sentence is only POS tagged and lemmatized.
        Format of the token tuple is (word, msd, head, deprel)."""

        tokens = []
        msd_input_tensor = []
        head_output_tensor = []

        # 1. Create the list of tokens and the MSD input vectors
        for i, (word, msd, head, deprel) in enumerate(sentence):
            tokens.append(word)
            msdv = self._msd.msd_reference_vector(msd)
            msdt = torch.tensor(msdv, dtype=torch.float32).view(1, -1)
            msd_input_tensor.append(msdt)

            if not runtime:
                # Here we have say 30 tokens to left/right of the current token i
                # head_vector will be 1 on the relative position of the head for token i
                # For PyTorch NLLLoss, we will keep the index of the correct (1) position
                # in the target (ground truth) vector
                center_index = RoDepParserTree._conf_head_window

                if head == 0:
                    head_output_tensor.append(center_index)
                else:
                    head_index = center_index + (head - (i + 1))

                    # If head index does not fit inside target vector, put
                    # 1 on the 0 index
                    if head_index < 0 or \
                        head_index >= 2 * RoDepParserTree._conf_head_window + 1:
                        print(stack()[
                              0][3] + f": found a head index out of limits at [{head_index}]!",
                              file=sys.stderr, flush=True)
                        head_index = 0
                    # end if

                    head_output_tensor.append(head_index)
                # end if
            # end if
        # end for

        msd_input_tensor = torch.cat(
            msd_input_tensor, dim=0).view(1, len(tokens), -1)

        if not runtime:
            # Shape of (1, len(sentence), feature_size)
            head_output_tensor = torch.tensor(
                head_output_tensor, dtype=torch.long).view(1, -1)
        # end if

        # 2. Do the BERT tokenization
        # Shape of outputs is (batch_size == 1, sub-token seq. len., 768)
        # Vector of [CLS] token is thus outputs[0,0,:]
        inputs = self._tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors='pt').to(_device)
        outputs = self._bertmodel(**inputs)

        # 3. This one maps from subtokens to tokens.
        token_ids = inputs.word_ids(batch_index=0)
        token_id_vectors = []
        token_vectors = []
        prev_tid = -1

        # 3.1 Skipping [CLS] and [SEP] subtokens
        # Doing averages of subtokens tensors for the given token
        for i in range(1, len(token_ids) - 1):
            tid = token_ids[i]
            tns = outputs.last_hidden_state[0, i, :]

            if prev_tid == -1 or tid == prev_tid:
                token_id_vectors.append(tns)
            else:
                # The average of the BERT outputs for
                # the subtokens of this token
                avg_tns = sum(token_id_vectors) / len(token_id_vectors)
                token_vectors.append(avg_tns.view(1, -1))
                token_id_vectors = [tns]
            # end if

            prev_tid = tid
        # end for

        avg_tns = sum(token_id_vectors) / len(token_id_vectors)
        token_vectors.append(avg_tns.view(1, -1))
        bert_input_tensor = torch.cat(token_vectors, dim=0).view(
            1, len(tokens), -1)
        
        if not runtime:
            return (bert_input_tensor, msd_input_tensor, head_output_tensor)
        else:
            return (bert_input_tensor, msd_input_tensor, None)
        # end if

    def _head_collate_fn(self, batch) -> Tuple[Tensor]:
        """This method will group sentences of the same length into a batch."""

        bert_tensor = []
        msd_tensor = []
        head_tensor = []
        prev_len = -1

        for snt in batch:
            if prev_len != -1 and prev_len != len(snt):
                break
            # end if

            btns, mtns, htns = self._process_sentence(sentence=snt)
            bert_tensor.append(btns)
            msd_tensor.append(mtns)
            head_tensor.append(htns)
            prev_len = len(snt)
        # end for

        bert_tensor = torch.cat(bert_tensor, dim=0).to(_device)
        msd_tensor = torch.cat(msd_tensor, dim=0).to(_device)
        head_tensor = torch.cat(head_tensor, dim=0).to(_device)

        return (bert_tensor, msd_tensor, head_tensor)

    def _do_one_epoch(self, epoch: int, dataloader: DataLoader):
        """Does one epoch of NN training, shuffling the examples first."""
        running_loss = 0.
        epoch_loss = []
        counter = 0

        for inputs_bert, inputs_msd, target_labels in tqdm(dataloader, desc=f'Epoch {epoch}'):
            counter += 1

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self._headmodel(x=(inputs_bert, inputs_msd))

            # Compute the loss and its gradients
            # Need to swap axes for NLLLoss
            outputs = torch.swapaxes(outputs, 1, 2)
            loss = self._loss_fn(outputs, target_labels)
            loss.backward()

            # Adjust learning weights and learning rate schedule
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if counter % 100 == 0:
                # Average loss per batch
                average_running_loss = running_loss / 100
                print(f'\n  -> batch {counter}/{len(dataloader)} loss: {average_running_loss:.5f}',
                      file=sys.stderr, flush=True)
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print(
            f'  -> average epoch {epoch} loss: {average_epoch_loss:.5f}', file=sys.stderr, flush=True)

    def train(self,
            train_sentences: List[List[Tuple]],
            dev_sentences: List[List[Tuple]], test_sentences: List[List[Tuple]]):
        """Does the head finder training."""

        self._tokenizer = AutoTokenizer.from_pretrained(
            RoDepParserTree._conf_bert)
        self._bertmodel = AutoModel.from_pretrained(
            RoDepParserTree._conf_bert)
        self._bertmodel.to(_device)
        self._loss_fn = nn.NLLLoss()
        self._optimizer = AdamW(self._headmodel.parameters(), lr=RoDepParserTree._conf_lr)
        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer, gamma=RoDepParserTree._conf_gamma_lr, verbose=True)
        train_dataset = DPDataset(sentences=train_sentences)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=4, shuffle=False, collate_fn=self._head_collate_fn)
        dev_dataloader = DataLoader(
            dataset=DPDataset(sentences=dev_sentences),
            batch_size=1, shuffle=False, collate_fn=self._head_collate_fn)
        test_dataloader = DataLoader(
            dataset=DPDataset(sentences=test_sentences),
            batch_size=1, shuffle=False, collate_fn=self._head_collate_fn)

        for ep in range(RoDepParserTree._conf_epochs):
            self._headmodel.train(True)
            self._bertmodel.train(True)
            self._do_one_epoch(epoch=ep + 1, dataloader=train_dataloader)
            self._headmodel.eval()
            self._bertmodel.eval()
            self.do_eval(dataloader=dev_dataloader, desc='dev')
            self._lr_scheduler.step()
            train_dataset.reshuffle()
        # end for

        self.do_eval(dataloader=test_dataloader, desc='test')
        self._save()

    def _save(self):
        torch_model_file = os.path.join(PARSER_MODEL_FOLDER, RoDepParserTree._conf_model_file)
        torch.save(self._headmodel.state_dict(), torch_model_file)
        self._tokenizer.save_pretrained(save_directory=PARSER1_TOKEN_MODEL_FOLDER)
        self._bertmodel.save_pretrained(save_directory=PARSER1_BERT_MODEL_FOLDER)

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            PARSER1_TOKEN_MODEL_FOLDER)
        self._bertmodel = AutoModel.from_pretrained(PARSER1_BERT_MODEL_FOLDER)
        self._bertmodel.to(_device)
        self._bertmodel.eval()
        torch_model_file = os.path.join(
            PARSER_MODEL_FOLDER, RoDepParserTree._conf_model_file)
        self._headmodel.load_state_dict(torch.load(
            torch_model_file, map_location=_device))
        # Put model into eval mode. It is only used for inferencing.
        self._headmodel.eval()

    def do_eval(self, dataloader: DataLoader, desc: str):
        correct = 0
        example_number = 0

        for inputs_bert, inputs_msd, target_labels in tqdm(dataloader, desc=f'Eval on {desc}set'):
            outputs = self._headmodel(x=(inputs_bert, inputs_msd))
            predicted_labels = torch.argmax(outputs, dim=2)
            found = (predicted_labels == target_labels)
            correct += found.sum().item()
            example_number += target_labels.shape[0] * target_labels.shape[1]
        # end for

        acc = correct / example_number
        print(f'Acc = {acc:.5f}', file=sys.stderr, flush=True)

    def parse_sentence(self, sentence: List[Tuple]) -> List[Tuple]:
        """This is the main entry into the Romanian depencency parser.
        Takes a POS tagged sentence (tokens are tuples of word, MSD, prob)
        and returns its parsed version."""

        parser_sentence = [(word, msd, 0, 'dep') for word, msd, prob in sentence]
        inputs_bert, inputs_msd, _ = self._process_sentence(
            sentence=parser_sentence, runtime=True)
        # Shape of outputs is [1, len(sentence), 2 * _conf_head_window + 1]
        outputs = self._headmodel(x=(inputs_bert, inputs_msd))
        # Shape of [1, len(sentence)].
        # Each [0, i] element is the best head index in the 2 * _conf_head_window + 1 sized vector.
        # Values are subtracted from the center of this vector, _conf_head_window, to mean offsets
        # left and right of i of i's head.
        #predicted_heads = torch.argmax(outputs, dim=2)
        
        parser_predicted_graph = {}
        best_root_id = 0
        best_root_prob = 0.0

        # Create the graph for the MST Chu-Liu/Edmonds algorithm
        for i in range(len(parser_sentence)):
            prob_heads_for_i = torch.exp(outputs[0, i, :])

            for k in range(prob_heads_for_i.shape[0]):
                pw = prob_heads_for_i[k].item()
                off = k - RoDepParserTree._conf_head_window
                j = i + off

                if j >= 0 and j < len(parser_sentence):
                    # Found a valid head, put it into the graph
                    di = i + 1
                    hj = j + 1
                    
                    if di == hj:
                        if pw > best_root_prob:
                            best_root_id = di
                            best_root_prob = pw
                        # end if
                    else:
                        if hj not in parser_predicted_graph:
                            parser_predicted_graph[hj] = {}
                        # end if

                        parser_predicted_graph[hj][di] = pw
                    # end if
                # end if
            # end for k
        # end for i

        parse_tree = chu_liu_edmonds(graph=parser_predicted_graph, root=best_root_id)
        
        # For each head in the tree
        for h in parse_tree:
            # For each dependent of the head in the tree
            for d in parse_tree[h]:
                parser_sentence[d - 1] = (
                    parser_sentence[d - 1][0],
                    parser_sentence[d - 1][1],
                    h,
                    parser_sentence[d - 1][3]
                )
            # end for
            
            # Set root
            if h == best_root_id:
                parser_sentence[h - 1] = (
                    parser_sentence[h - 1][0],
                    parser_sentence[h - 1][1],
                    0,
                    parser_sentence[h - 1][3]
                )
            # end if
        # end for

        return parser_sentence
