from typing import List, Tuple, Set, Union
import os
import sys
from random import shuffle
from tqdm import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from utils.MSD import MSD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from . import _device
from config import PARSER_MODEL_FOLDER, \
    PARSER2_BERT_MODEL_FOLDER, PARSER2_TOKEN_MODEL_FOLDER


class RoBERTDepRelFinder(nn.Module):
    """This class computes the probability distribution of
    a sequence of dependency relation labels along a tree path
    from the root to a leaf."""

    _conf_lstm_size = 1024
    _conf_drop_prob = 0.33

    def __init__(self, msd_size: int, deprel_size: int):
        """`msd_size` - the size of the MSD vector, from MSD.msd_reference_vector()
        `head_window_size`: how much to go left/right from current token to search for its head
        `name_or_path`: the BERT model HF name/saved folder"""

        super().__init__()

        self._nonlin_fn = nn.LeakyReLU()
        self._drop = nn.Dropout(p=RoBERTDepRelFinder._conf_drop_prob)
        self._logsoftmax = nn.LogSoftmax(dim=2)

        self._nn_lstm = nn.LSTM(
            input_size=768, hidden_size=RoBERTDepRelFinder._conf_lstm_size,
            num_layers=1, batch_first=True, bidirectional=False)

        linear_input_size_0 = msd_size + RoBERTDepRelFinder._conf_lstm_size
        linear_input_size_1 = linear_input_size_0 // 2

        self._nn_linear_0 = nn.Linear(
            in_features=linear_input_size_0,
            out_features=linear_input_size_1, dtype=torch.float32)
        self._nn_linear_1 = nn.Linear(
            in_features=linear_input_size_1,
            out_features=deprel_size, dtype=torch.float32)

        self.to(_device)

    def forward(self, x):
        x_b, x_m = x
        b_size = x_b.shape[0]
        h_0 = torch.zeros(
            1, b_size,
            RoBERTDepRelFinder._conf_lstm_size).to(device=_device)
        c_0 = torch.zeros(
            1, b_size,
            RoBERTDepRelFinder._conf_lstm_size).to(device=_device)

        the_output, (h_n, c_n) = self._nn_lstm(x_b, (h_0, c_0))
        the_output = torch.cat([the_output, x_m], dim=2).to(device=_device)
        the_output = self._nn_linear_0(the_output)
        the_output = self._nonlin_fn(the_output)
        the_output = self._drop(the_output)
        the_output = self._nn_linear_1(the_output)
        the_output = self._logsoftmax(the_output)

        return the_output


class DPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self._paths_by_length = {}

    def add_sentence_paths(self, sentence: List[Tuple], paths: List[List[int]]):
        for path in paths:
            pathlen = len(path)

            if pathlen not in self._paths_by_length:
                self._paths_by_length[pathlen] = []
            # end if

            self._paths_by_length[pathlen].append((path, sentence))
        # end for
    # end def

    def reshuffle(self):
        path_lengths = list(self._paths_by_length.keys())
        path_lengths = sorted(path_lengths, reverse=True)
        self._paths = []

        for pathlen in path_lengths:
            shuffle(self._paths_by_length[pathlen])
            self._paths.extend(self._paths_by_length[pathlen])
        # end for
    # end def

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index):
        return self._paths[index]


class RoDepParserLabel(object):
    """This class uses BERT to assign label names to sequences of dependency
    relations, starting from root to a tree leaf. Words in a dependency relation
    are described by their BERT embeddings and MSDs in context."""

    _conf_model_file = 'modeltwo.pt'
    _conf_bert = 'dumitrescustefan/bert-base-romanian-cased-v1'
    # Initial learning rate
    _conf_lr = 5e-5
    # Multiplying factor between epochs of the LR
    _conf_gamma_lr = 0.9
    _conf_epochs = 3

    def __init__(self, msd: MSD, deprels: Set[str]):
        """Takes the MSD description object `msd` and the set of
        all possible dependency relations `deprels`."""

        self._msd = msd
        # This is the set of all possible dependency relations
        self._deprels = sorted(list(deprels))
        self._deprelmodel = RoBERTDepRelFinder(
            msd_size=self._msd.get_output_vector_size(),
            deprel_size=len(self._deprels))

    def _process_path(self, sentence: List[Tuple], path: List[int], runtime: bool = False) -> Tuple[Union[Tensor, None]]:
        """Takes a path and a sentence and produces tensors for the path
        from the root to the leaf."""

        tokens = []

        # 1. Find the root of the sentence
        for i, (word, msd, head, deprel) in enumerate(sentence):
            tokens.append(word)
        # end for

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
                # Shape of (1, 768)
                token_vectors.append(avg_tns.view(1, -1))
                token_id_vectors = [tns]
            # end if

            prev_tid = tid
        # end for

        avg_tns = sum(token_id_vectors) / len(token_id_vectors)
        token_vectors.append(avg_tns.view(1, -1))

        # 4. Create input/output tensors for root-to-leaf path
        bert_input_tensor = []
        msd_input_tensor = []
        deprel_output_tensor = []

        for i in path:
            msd = sentence[i][1]
            msdv = self._msd.msd_reference_vector(msd)
            msdt = torch.tensor(msdv, dtype=torch.float32).view(1, -1)
            msd_input_tensor.append(msdt)
            bert_input_tensor.append(token_vectors[i])

            if not runtime:
                dr = sentence[i][3]
                deprel_output_tensor.append(self._deprels.index(dr))
            # end if
        # end for

        bert_input_tensor = torch.cat(bert_input_tensor, dim=0).view(
            1, len(path), -1)
        msd_input_tensor = torch.cat(
            msd_input_tensor, dim=0).view(1, len(path), -1)
        
        if not runtime:
            deprel_output_tensor = \
                torch.tensor(deprel_output_tensor,
                                dtype=torch.long)
        # end if

        if not runtime:
            return bert_input_tensor, msd_input_tensor, deprel_output_tensor
        else:
            return bert_input_tensor, msd_input_tensor, None
        # end if

    def _deprel_collate_fn(self, batch) -> Tuple[Tensor]:
        """This method will group sentence paths of the same length into a batch."""

        bert_tensor = []
        msd_tensor = []
        drel_tensor = []
        prev_len = -1

        for pth, snt in batch:
            if prev_len != -1 and prev_len != len(pth):
                break
            # end if

            btns, mtns, drtns = self._process_path(sentence=snt, path=pth)
            bert_tensor.append(btns)
            msd_tensor.append(mtns)
            drel_tensor.append(drtns.view(1, -1))
            prev_len = len(pth)
        # end for

        bert_tensor = torch.cat(bert_tensor, dim=0).to(_device)
        msd_tensor = torch.cat(msd_tensor, dim=0).to(_device)
        drel_tensor = torch.cat(drel_tensor, dim=0).to(_device)

        return (bert_tensor, msd_tensor, drel_tensor)

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
            outputs = self._deprelmodel(x=(inputs_bert, inputs_msd))

            # Compute the loss and its gradients
            # Need to swap axes for NLLLoss
            outputs = torch.swapaxes(outputs, 1, 2)
            loss = self._loss_fn(outputs, target_labels)
            loss.backward()

            # Adjust learning weights and learning rate schedule
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if counter % 200 == 0:
                # Average loss per batch
                average_running_loss = running_loss / 200
                print(f'\n  -> batch {counter}/{len(dataloader)} loss: {average_running_loss:.5f}',
                      file=sys.stderr, flush=True)
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print(
            f'  -> average epoch {epoch} loss: {average_epoch_loss:.5f}', file=sys.stderr, flush=True)

    def _depth_first_paths(self,
        sentence: List[Tuple], root: int,
        all_paths: List[List[int]], crt_path: List[int]):
        """Finds all possible paths from the root to any of its leaves."""

        crt_path.append(root)
        root_children = []

        for i, (word, msd, head, deprel) in enumerate(sentence):
            if head == root + 1:
                root_children.append(i)
            # end if
        # end for

        for ch in root_children:
            self._depth_first_paths(sentence, ch, all_paths, crt_path)
        # end for

        if not root_children:
            # Test leaf condition to append a path
            all_paths.append(list(crt_path))
        # end

        crt_path.pop()

    def train(self,
              train_sentences: List[List[Tuple]],
              dev_sentences: List[List[Tuple]], test_sentences: List[List[Tuple]]):
        """Does the dependency relation labeling training."""

        self._tokenizer = AutoTokenizer.from_pretrained(
            RoDepParserLabel._conf_bert)
        self._bertmodel = AutoModel.from_pretrained(
            RoDepParserLabel._conf_bert)
        self._bertmodel.to(_device)

        train_dataset = self._create_dataset(sentences=train_sentences, desc='train')
        train_dataset.reshuffle()
        dev_dataset = self._create_dataset(sentences=dev_sentences, desc='dev')
        dev_dataset.reshuffle()
        test_dataset = self._create_dataset(sentences=test_sentences, desc='test')
        test_dataset.reshuffle()

        self._loss_fn = nn.NLLLoss()
        self._optimizer = AdamW(
            self._deprelmodel.parameters(), lr=RoDepParserLabel._conf_lr)
        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer, gamma=RoDepParserLabel._conf_gamma_lr, verbose=True)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=16, shuffle=False, collate_fn=self._deprel_collate_fn)
        dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=1, shuffle=False, collate_fn=self._deprel_collate_fn)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1, shuffle=False, collate_fn=self._deprel_collate_fn)

        for ep in range(RoDepParserLabel._conf_epochs):
            self._deprelmodel.train(True)
            self._bertmodel.train(True)
            self._do_one_epoch(epoch=ep + 1, dataloader=train_dataloader)
            self._deprelmodel.eval()
            self._bertmodel.eval()
            self.do_eval(dataloader=dev_dataloader, desc='dev')
            self._lr_scheduler.step()
            train_dataset.reshuffle()
        # end for

        self.do_eval(dataloader=test_dataloader, desc='test')
        self._save()

    def _save(self):
        torch_model_file = os.path.join(
            PARSER_MODEL_FOLDER, RoDepParserLabel._conf_model_file)
        torch.save(self._deprelmodel.state_dict(), torch_model_file)
        self._tokenizer.save_pretrained(
            save_directory=PARSER2_TOKEN_MODEL_FOLDER)
        self._bertmodel.save_pretrained(
            save_directory=PARSER2_BERT_MODEL_FOLDER)

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            PARSER2_TOKEN_MODEL_FOLDER)
        self._bertmodel = AutoModel.from_pretrained(PARSER2_BERT_MODEL_FOLDER)
        self._bertmodel.to(_device)
        self._bertmodel.eval()
        torch_model_file = os.path.join(
            PARSER_MODEL_FOLDER, RoDepParserLabel._conf_model_file)
        self._deprelmodel.load_state_dict(torch.load(
            torch_model_file, map_location=_device))
        # Put model into eval mode. It is only used for inferencing.
        self._deprelmodel.eval()

    def do_eval(self, dataloader: DataLoader, desc: str):
        correct = 0
        example_number = 0

        for inputs_bert, inputs_msd, target_labels in tqdm(dataloader, desc=f'Eval on {desc}set'):
            outputs = self._deprelmodel(x=(inputs_bert, inputs_msd))
            predicted_labels = torch.argmax(outputs, dim=2)
            found = (predicted_labels == target_labels)
            correct += found.sum().item()
            example_number += target_labels.shape[0] * target_labels.shape[1]
        # end for

        acc = correct / example_number
        print(f'Acc = {acc:.5f}', file=sys.stderr, flush=True)

    def find_sentence_paths(self, sentence: List[Tuple]) -> List[List[int]]:
        """Takes a sentence and extracts all paths from root to leaves."""

        root = 0

        for i, (word, msd, head, deprel) in enumerate(sentence):
            if head == 0:
                root = i
                break
            # end if
        # end for

        # root_to_leaf contains all possible paths from root to all leaves
        # Indexes are 0-based, can be used in the sentence, directly
        root_to_leaf = []
        self._depth_first_paths(sentence, root, root_to_leaf, [])

        return root_to_leaf

    def _create_dataset(self, sentences: List[List[Tuple]], desc: str) -> DPDataset:
        dataset = DPDataset()

        for snt in tqdm(sentences, desc=f'Creating {desc}set for DR'):
            r2l = self.find_sentence_paths(sentence=snt)
            dataset.add_sentence_paths(sentence=snt, paths=r2l)
        # end for

        return dataset
