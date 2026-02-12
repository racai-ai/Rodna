from typing import List, Tuple, Union
import os
from glob import glob
from random import shuffle
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from .lexicon import MSD
from .bert_model import RoBERTModel, _device, dumitrescu_bert_v1
from .tokenizer import RoTokenizer
from ..utils.mst import chu_liu_edmonds
from . import _device
from .. import logger
from ..config import PARSER_MODEL_FOLDER, PARSER1_BERT_MODEL_FOLDER


class RoBERTHeadFinder(nn.Module):
    """This class finds the probability distribution of possible heads
    for the current token. Uses a MLM BERT model for word embeddings
    which is fine-tuned to find head information."""

    _conf_lstm_size = 1024
    _conf_drop_prob = 0.25

    def __init__(self, msd_size: int, head_window_size: int, embed_size: int):
        """`msd_size` - the size of the MSD vector, from MSD.msd_reference_vector()
        `head_window_size`: how much to go left/right from current token to search for its head
        `embed_size`: the size of the BERT hidden state"""

        super().__init__()

        self._nonlin_fn = nn.LeakyReLU()
        self._drop = nn.Dropout(p=RoBERTHeadFinder._conf_drop_prob)
        self._logsoftmax = nn.LogSoftmax(dim=2)

        self._nn_lstm = nn.LSTM(
            input_size=embed_size, hidden_size=RoBERTHeadFinder._conf_lstm_size,
            num_layers=1, batch_first=True, bidirectional=True)

        linear_io_size = msd_size + 2 * RoBERTHeadFinder._conf_lstm_size

        self._nn_linear = nn.Linear(
            in_features=linear_io_size,
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
        the_output = self._drop(the_output)
        the_output = self._nn_linear(the_output)
        the_output = self._logsoftmax(the_output)

        return the_output


class HeadDataset(Dataset):
    """This is a dependency parser dataset. Sorts the sentences by length
    so that they can form batches for LSTMs/GRUs. Shuffles the lengths so that
    we have a shuffled dataset."""

    def __init__(self, sentences: List[List[Tuple[str, str, int, str]]]):
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
        sentence_lengths.sort(reverse=True)
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
    # Take 2 * head window + 1 for the output vector dimension
    _conf_head_window = 70
    _conf_epochs = 20
    _all_parser_model_files = [
        os.path.join(PARSER_MODEL_FOLDER, _conf_model_file),
        os.path.join(PARSER1_BERT_MODEL_FOLDER, '*.json'),
        os.path.join(PARSER1_BERT_MODEL_FOLDER, 'model.safetensors'),
        os.path.join(PARSER1_BERT_MODEL_FOLDER, 'vocab.txt')
    ]

    def __init__(self, msd: MSD, tok: RoTokenizer) -> None:
        super().__init__()
        self._msd = msd
        self._tokenizer = tok

    def _process_sentence(self, sentence: List[Tuple], runtime: bool = False) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
        """Gets a sentence and returns a tuple of tensors to be run
        through the neural network, as part of a batch.
        Shape of tuple of inputs is [1, len(sentence), feature_size].
        When `runtime is True`, the sentence is only POS tagged and lemmatized.
        Format of the token tuple is (word, msd, head, deprel)."""

        tokens = []
        msd_input_tensor = []
        head_output_tensor = []
        bert_tokens = []

        # 1. Create the list of tokens and the MSD input vectors
        for i, (word, msd, head, deprel) in enumerate(sentence):
            tokens.append(word)
            bert_tokens.append((word, self._tokenizer.tag_word(word)))
            msdv = self._msd.msd_reference_vector(msd)
            msdt = torch.tensor(msdv, dtype=torch.float32).view(1, -1)
            msd_input_tensor.append(msdt)

            if not runtime:
                # Here we have say 30 tokens to left/right of the current token i
                # head_vector will be 1 on the relative position of the head for token i
                # For PyTorch NLLLoss, we will keep the index of the correct (1) position
                # in the target (ground truth) vector
                if head == 0:
                    head_output_tensor.append(
                        RoDepParserTree._conf_head_window)
                else:
                    head_index = RoDepParserTree._conf_head_window + \
                        (head - (i + 1))

                    # If head index does not fit inside target vector, put
                    # 1 on the 0 index
                    if head_index < 0 or \
                        head_index >= 2 * RoDepParserTree._conf_head_window + 1:
                        logger.debug(f"Found a head index out of limits at [{head_index}]!")
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

        # 2. Do the BERT embeddings computation
        # Shape of outputs is (batch_size == 1, sub-token seq. len., 768)
        sentence_embeddings = [t.view(1, -1)
                               for t in self._ro_model.bert_embeddings(tokens=bert_tokens)]
        bert_input_tensor = torch.cat(sentence_embeddings, dim=0)
        bert_input_tensor = bert_input_tensor.view(
            1, bert_input_tensor.shape[0], bert_input_tensor.shape[1])
        
        if not runtime:
            return bert_input_tensor, msd_input_tensor, head_output_tensor
        else:
            return bert_input_tensor.to(_device), msd_input_tensor.to(_device), None
        # end if

    def _trim_batch(self, batch: List[List[Tuple[str, str, int, str]]]) -> List[List[Tuple[str, str, int, str]]]:
        lowest_dim = 1_000_000

        for snt in batch:
            if len(snt) < lowest_dim:
                lowest_dim = len(snt)
            # end if
        # end for

        trimmed_batch = []

        for b in batch:
            trimmed_batch.append(b[:lowest_dim])
        # end for

        return trimmed_batch

    def _head_collate_fn(self, batch: List[List[Tuple[str, str, int, str]]]) -> Tuple[Tensor, Tensor, Tensor]:
        """This method will group sentences of the same length into a batch."""

        batch = self._trim_batch(batch=batch)

        bert_tensor = []
        msd_tensor = []
        head_tensor = []

        for snt in batch:
            btns, mtns, htns = self._process_sentence(sentence=snt)
            bert_tensor.append(btns)
            msd_tensor.append(mtns)
            head_tensor.append(htns)
        # end for

        bert_tensor = torch.cat(bert_tensor, dim=0).to(_device)
        msd_tensor = torch.cat(msd_tensor, dim=0).to(_device)
        head_tensor = torch.cat(head_tensor, dim=0).to(_device)

        return bert_tensor, msd_tensor, head_tensor

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
                logger.info(f'Batch {counter}/{len(dataloader)} loss: {average_running_loss:.5f}')
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        logger.info(f'Average epoch {epoch} loss: {average_epoch_loss:.5f}')

    def train(self,
            train_sentences: List[List[Tuple]],
            dev_sentences: List[List[Tuple]], test_sentences: List[List[Tuple]]):
        """Does the head finder training."""

        self._ro_model = RoBERTModel(path_or_name=dumitrescu_bert_v1,
                                     fine_tune=True)
        self._headmodel = RoBERTHeadFinder(
            msd_size=self._msd.get_output_vector_size(),
            head_window_size=RoDepParserTree._conf_head_window,
            embed_size=self._ro_model.get_embedding_size())

        self._loss_fn = nn.NLLLoss()

        # Don’t decay LayerNorm & biases
        bert_decay_params, bert_no_decay_params = [], []

        for name, param in self._ro_model.bert_model.named_parameters():
            if param.ndim == 1 or name.endswith(".bias"):
                bert_no_decay_params.append(param)
            else:
                bert_decay_params.append(param)
            # end if
        # end for

        self._optimizer = AdamW([
            {"params": bert_decay_params, "weight_decay": 0.01, "lr": 5e-5},
            {"params": bert_no_decay_params, "weight_decay": 0.0, "lr": 5e-5},
            {"params": self._headmodel.parameters(), "weight_decay": 0.0, "lr": 1e-3}]
        )
        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer, gamma=0.98)

        train_dataset = HeadDataset(sentences=train_sentences)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=8, shuffle=False, collate_fn=self._head_collate_fn)
        dev_dataloader = DataLoader(
            dataset=HeadDataset(sentences=dev_sentences),
            batch_size=1, shuffle=False, collate_fn=self._head_collate_fn)
        test_dataloader = DataLoader(
            dataset=HeadDataset(sentences=test_sentences),
            batch_size=1, shuffle=False, collate_fn=self._head_collate_fn)

        best_acc = 0.

        for ep in range(RoDepParserTree._conf_epochs):
            self._headmodel.train()
            self._ro_model.bert_model.train()
            self._do_one_epoch(epoch=ep + 1, dataloader=train_dataloader)
            self._headmodel.eval()
            self._ro_model.bert_model.eval()
            ep_acc = self.do_eval(dataloader=dev_dataloader, ml_type='dev')

            if ep_acc > best_acc:
                logger.info(f'Saving better RoDepParserTree model with Acc = {ep_acc:.5f}')
                best_acc = ep_acc
                self._delete_parser_files()
                self._save()
            # end if

            self._lr_scheduler.step()
            bert_lr, _, head_lr = self._lr_scheduler.get_last_lr()
            logger.info(f'Setting new BERT LR to [{bert_lr:.7f}]')
            logger.info(f'Setting new head model LR to [{head_lr:.7f}]')

            train_dataset.reshuffle()
        # end for

        self.do_eval(dataloader=test_dataloader, ml_type='test')

    def _save(self):
        torch_model_file = os.path.join(PARSER_MODEL_FOLDER, RoDepParserTree._conf_model_file)
        torch.save(self._headmodel.state_dict(), torch_model_file)
        self._ro_model.save(destination_folder=PARSER1_BERT_MODEL_FOLDER)

    def load(self):
        self._ro_model = RoBERTModel(path_or_name=PARSER1_BERT_MODEL_FOLDER)
        self._ro_model.bert_model.eval()
        self._headmodel = RoBERTHeadFinder(
            msd_size=self._msd.get_output_vector_size(),
            head_window_size=RoDepParserTree._conf_head_window,
            embed_size=self._ro_model.get_embedding_size())
        torch_model_file = os.path.join(
            PARSER_MODEL_FOLDER, RoDepParserTree._conf_model_file)
        self._headmodel.load_state_dict(torch.load(
            torch_model_file, map_location=_device))
        # Put model into eval mode. It is only used for inferencing.
        self._headmodel.eval()

    def _delete_parser_files(self):
        for file in RoDepParserTree._all_parser_model_files:
            if os.path.isfile(file):
                logger.info(f'Removing RoDepParserTree model file [{file}]')
                os.remove(file)
            else:
                # Wildcard file spec
                for path in glob(file):
                    logger.info(
                        f'Removing RoDepParserTree model file [{path}]')
                    os.remove(path)
                # end for
            # end if
        # end for

    def do_eval(self, dataloader: DataLoader, ml_type: str) -> float:
        correct = 0
        example_number = 0

        for inputs_bert, inputs_msd, target_labels in tqdm(dataloader, desc=f'Eval on [{ml_type}] set'):
            outputs = self._headmodel(x=(inputs_bert, inputs_msd))
            predicted_labels = torch.argmax(outputs, dim=2)
            found = (predicted_labels == target_labels)
            correct += found.sum().item()
            example_number += target_labels.shape[0] * target_labels.shape[1]
        # end for

        acc = correct / example_number
        logger.info(f'Acc = {acc:.5f}')

        return acc

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
        # Return the link probability, as well
        for h in parse_tree:
            # For each dependent of the head in the tree
            for d in parse_tree[h]:
                parser_sentence[d - 1] = (
                    parser_sentence[d - 1][0],
                    parser_sentence[d - 1][1],
                    h,
                    parser_sentence[d - 1][3],
                    parse_tree[h][d]
                )
            # end for
            
            # Set root
            if h == best_root_id:
                parser_sentence[h - 1] = (
                    parser_sentence[h - 1][0],
                    parser_sentence[h - 1][1],
                    0,
                    parser_sentence[h - 1][3],
                    best_root_prob
                )
            # end if
        # end for

        return parser_sentence
