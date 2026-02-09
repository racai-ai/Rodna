from typing import List, Tuple, Set, Union
import os
from glob import glob
from random import shuffle
from tqdm import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset
from .lexicon import MSD
from .bert_model import RoBERTModel, dumitrescu_bert_v1
from .tokenizer import RoTokenizer
from . import PARSER_MODEL_FOLDER, \
    PARSER2_BERT_MODEL_FOLDER, _device, logger


class RoBERTDepRelFinder(nn.Module):
    """This class computes the probability distribution of
    a sequence of dependency relation labels along a tree path
    from the root to a leaf."""

    _conf_rnn_size = 1024
    _conf_drop_prob = 0.25

    def __init__(self, msd_size: int, deprel_size: int, embed_size: int):
        """`msd_size` - the size of the MSD vector, from MSD.msd_reference_vector()
        `deprel_size`: the size of the dependency relation set
        `embed_size`: the size of the BERT hidden state"""

        super().__init__()

        self._nonlin_fn = nn.LeakyReLU()
        self._drop = nn.Dropout(p=RoBERTDepRelFinder._conf_drop_prob)
        self._logsoftmax = nn.LogSoftmax(dim=2)

        self._nn_rnn = nn.GRU(
            input_size=embed_size, hidden_size=RoBERTDepRelFinder._conf_rnn_size,
            num_layers=1, batch_first=True, bidirectional=False)

        linear_input_size = msd_size + RoBERTDepRelFinder._conf_rnn_size

        self._nn_linear = nn.Linear(
            in_features=linear_input_size,
            out_features=deprel_size, dtype=torch.float32)

        self.to(_device)

    def forward(self, x):
        x_b, x_m = x
        b_size = x_b.shape[0]
        h_0 = torch.zeros(
            1, b_size,
            RoBERTDepRelFinder._conf_rnn_size).to(device=_device)

        the_output, h_n = self._nn_rnn(x_b, h_0)
        the_output = torch.cat([the_output, x_m], dim=2).to(device=_device)
        the_output = self._drop(the_output)
        the_output = self._nn_linear(the_output)
        the_output = self._logsoftmax(the_output)

        return the_output


class PathDataset(Dataset):
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
        path_lengths.sort(reverse=True)
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
    _conf_epochs = 10
    _all_parser_model_files = [
        os.path.join(PARSER_MODEL_FOLDER, _conf_model_file),
        os.path.join(PARSER2_BERT_MODEL_FOLDER, '*.json'),
        os.path.join(PARSER2_BERT_MODEL_FOLDER, 'model.safetensors'),
        os.path.join(PARSER2_BERT_MODEL_FOLDER, 'vocab.txt')
    ]

    def __init__(self, msd: MSD, tok: RoTokenizer, deprels: Set[str]):
        """Takes the MSD description object `msd` and the set of
        all possible dependency relations `deprels`."""

        self._msd = msd
        # This is the set of all possible dependency relations
        self._deprels = sorted(list(deprels))
        self._tokenizer = tok

    def _process_path(self, sentence: List[Tuple], path: List[int],
                      runtime: bool = False) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
        """Takes a path and a sentence and produces tensors for the path
        from the root to the leaf."""

        tokens = []
        bert_tokens = []

        # 1. Find the root of the sentence
        for i, (word, msd, head, deprel) in enumerate(sentence):
            tokens.append(word)
            bert_tokens.append((word, self._tokenizer.tag_word(word)))
        # end for

        # 2. Do the BERT tokenization
        # Shape of outputs is (batch_size == 1, sub-token seq. len., 768)
        # Vector of [CLS] token is thus outputs[0,0,:]
        sentence_embeddings = [t.view(1, -1)
                               for t in self._ro_model.bert_embeddings(tokens=bert_tokens)]

        # 3. Create input/output tensors for root-to-leaf path
        bert_input_tensor = []
        msd_input_tensor = []
        deprel_output_tensor = []

        for i in path:
            msd = sentence[i][1]
            msdv = self._msd.msd_reference_vector(msd)
            msdt = torch.tensor(msdv, dtype=torch.float32).view(1, -1)
            msd_input_tensor.append(msdt)
            bert_input_tensor.append(sentence_embeddings[i])

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
            # Here the collate fn is taking care of moving tensors to cuda:0
            return bert_input_tensor, msd_input_tensor, deprel_output_tensor
        else:
            return bert_input_tensor.to(_device), msd_input_tensor.to(_device), None
        # end if

    def _trim_batch(self, batch: List[Tuple[List[int], List[Tuple]]]) -> List[Tuple[List[int], List[Tuple]]]:
        lowest_dim = 1_000_000

        for pth, snt in batch:
            if len(pth) < lowest_dim:
                lowest_dim = len(pth)
            # end if
        # end for

        trimmed_batch = []

        for pth, snt in batch:
            trimmed_batch.append((pth[:lowest_dim], snt))
        # end for

        return trimmed_batch

    def _deprel_collate_fn(self, batch: List[Tuple[List[int], List[Tuple]]]) -> Tuple[Tensor, Tensor, Tensor]:
        """This method will group sentence paths of the same length into a batch."""

        batch = self._trim_batch(batch=batch)

        bert_tensor = []
        msd_tensor = []
        drel_tensor = []

        for pth, snt in batch:
            btns, mtns, drtns = self._process_path(sentence=snt, path=pth)
            bert_tensor.append(btns)
            msd_tensor.append(mtns)
            drel_tensor.append(drtns.view(1, -1))
        # end for

        bert_tensor = torch.cat(bert_tensor, dim=0).to(_device)
        msd_tensor = torch.cat(msd_tensor, dim=0).to(_device)
        drel_tensor = torch.cat(drel_tensor, dim=0).to(_device)

        return bert_tensor, msd_tensor, drel_tensor

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

            if counter % 100 == 0:
                # Average loss per batch
                average_running_loss = running_loss / 100
                logger.info(f'Batch {counter}/{len(dataloader)} loss: {average_running_loss:.7f}')
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        logger.info(f'Average epoch {epoch} loss: {average_epoch_loss:.7f}')

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

        self._ro_model = RoBERTModel(path_or_name=dumitrescu_bert_v1,
                                     fine_tune=True)
        self._deprelmodel = RoBERTDepRelFinder(
            msd_size=self._msd.get_output_vector_size(),
            deprel_size=len(self._deprels),
            embed_size=self._ro_model.get_embedding_size())

        train_dataset = self._create_dataset(sentences=train_sentences, desc='train')
        train_dataset.reshuffle()
        dev_dataset = self._create_dataset(sentences=dev_sentences, desc='dev')
        dev_dataset.reshuffle()
        test_dataset = self._create_dataset(sentences=test_sentences, desc='test')
        test_dataset.reshuffle()

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
            {"params": bert_decay_params, "weight_decay": 0.01, "lr": 2e-5},
            {"params": bert_no_decay_params, "weight_decay": 0.0, "lr": 2e-5},
            {"params": self._deprelmodel.parameters(), "weight_decay": 0.0, "lr": 1e-3}]
        )
        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer, gamma=0.95)
        
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=32, shuffle=False, collate_fn=self._deprel_collate_fn)
        dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=1, shuffle=False, collate_fn=self._deprel_collate_fn)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1, shuffle=False, collate_fn=self._deprel_collate_fn)

        best_acc = 0.

        for ep in range(RoDepParserLabel._conf_epochs):
            self._deprelmodel.train()
            self._ro_model.bert_model.train()
            self._do_one_epoch(epoch=ep + 1, dataloader=train_dataloader)
            self._deprelmodel.eval()
            self._ro_model.bert_model.eval()
            ep_acc = self.do_eval(dataloader=dev_dataloader, ml_type='dev')

            if ep_acc > best_acc:
                logger.info(f'Saving better RoDepParserLabel model with Acc = {ep_acc:.5f}')
                best_acc = ep_acc
                self._delete_parser_files()
                self._save()
            # end if

            self._lr_scheduler.step()
            bert_lr, _, deprel_lr = self._lr_scheduler.get_last_lr()
            logger.info(f'Setting new BERT LR to [{bert_lr:.7f}]')
            logger.info(f'Setting new dependency relations model LR to [{deprel_lr:.7f}]')

            train_dataset.reshuffle()
        # end for

        self.do_eval(dataloader=test_dataloader, ml_type='test')

    def _delete_parser_files(self):
        for file in RoDepParserLabel._all_parser_model_files:
            if os.path.isfile(file):
                logger.info(f'Removing RoDepParserLabel model file [{file}]')
                os.remove(file)
            else:
                # Wildcard file spec
                for path in glob(file):
                    logger.info(
                        f'Removing RoDepParserLabel model file [{path}]')
                    os.remove(path)
                # end for
            # end if
        # end for

    def _save(self):
        torch_model_file = os.path.join(
            PARSER_MODEL_FOLDER, RoDepParserLabel._conf_model_file)
        torch.save(self._deprelmodel.state_dict(), torch_model_file)
        self._ro_model.save(destination_folder=PARSER2_BERT_MODEL_FOLDER)

    def load(self):
        self._ro_model = RoBERTModel(path_or_name=PARSER2_BERT_MODEL_FOLDER)
        self._ro_model.bert_model.eval()
        self._deprelmodel = RoBERTDepRelFinder(
            msd_size=self._msd.get_output_vector_size(),
            deprel_size=len(self._deprels),
            embed_size=self._ro_model.get_embedding_size())
        torch_model_file = os.path.join(
            PARSER_MODEL_FOLDER, RoDepParserLabel._conf_model_file)
        self._deprelmodel.load_state_dict(torch.load(
            torch_model_file, map_location=_device))
        # Put model into eval mode. It is only used for inferencing.
        self._deprelmodel.eval()

    def do_eval(self, dataloader: DataLoader, ml_type: str) -> float:
        correct = 0
        example_number = 0

        for inputs_bert, inputs_msd, target_labels in tqdm(dataloader, desc=f'Eval on [{ml_type}] set'):
            outputs = self._deprelmodel(x=(inputs_bert, inputs_msd))
            predicted_labels = torch.argmax(outputs, dim=2)
            found = (predicted_labels == target_labels)
            correct += found.sum().item()
            example_number += target_labels.shape[0] * target_labels.shape[1]
        # end for

        acc = correct / example_number
        logger.info(f'Acc = {acc:.5f}')

        return acc

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

    def label_path(self, sentence: List[Tuple], path: List[int]) -> List[Tuple]:
        inputs_bert, inputs_msd, _ = self._process_path(sentence, path, runtime=True)

        with torch.inference_mode():
            outputs = self._deprelmodel(x=(inputs_bert, inputs_msd))
        # end with

        # To get probabilities
        outputs = torch.exp(outputs)
        predicted_labels = torch.argmax(outputs, dim=2)
        result = []

        for i in range(predicted_labels.shape[1]):
            dri = predicted_labels[0, i].item()
            dr = self._deprels[dri]
            drp = outputs[0, i, dri].item()

            result.append((dr, drp))
        # end for

        return result

    def _create_dataset(self, sentences: List[List[Tuple]], desc: str) -> PathDataset:
        dataset = PathDataset()

        for snt in tqdm(sentences, desc=f'RoDepParserLabel [{desc}] set'):
            r2l = self.find_sentence_paths(sentence=snt)
            dataset.add_sentence_paths(sentence=snt, paths=r2l)
        # end for

        return dataset
