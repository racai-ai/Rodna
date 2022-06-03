import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import RMSprop
from torch.nn.utils.rnn import pack_sequence
from tqdm import tqdm
from inspect import stack
from random import shuffle
from utils.errors import print_error
from utils.Lex import Lex
from config import TBL_WORDFORM_FILE, \
    ROINFLECT_MODEL_FOLDER, ROINFLECT_CHARID_FILE, \
    ROINFLECT_CACHE_FILE

torch.manual_seed(1234)
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RoInflectModule(nn.Module):
    """PyTorch neural net for encoding the characters of a word
    using a LSTM neural network."""

    _conf_char_embed_size = 32
    _conf_lstm_size = 256
    _conf_dense_size = 512

    def __init__(self, char_vocab_dim: int, msd_vector_dim: int):
        super().__init__()
        self._layer_embed = nn.Embedding(
            num_embeddings=char_vocab_dim,
            embedding_dim=RoInflectModule._conf_char_embed_size
        )
        self._layer_lstm = nn.LSTM(
            input_size=RoInflectModule._conf_char_embed_size,
            hidden_size=RoInflectModule._conf_lstm_size,
            batch_first=True)
        self._layer_linear_1 = nn.Linear(
            in_features=RoInflectModule._conf_lstm_size,
            out_features=RoInflectModule._conf_dense_size
        )
        self._layer_linear_2 = nn.Linear(
            in_features=RoInflectModule._conf_dense_size,
            out_features=msd_vector_dim
        )
        self._layer_drop = nn.Dropout(p=0.3)
        self._tanh = nn.Tanh()
        self._sigmoid = nn.Sigmoid()
        self.to(device=_device)

    def forward(self, x):
        b_size = len(x)
        batch_sequences = []

        # Compute embeddings first, one by one...
        for xw in x:
            batch_sequences.append(self._layer_embed(xw))
        # end for

        packed_embeddings = pack_sequence(batch_sequences, enforce_sorted=False)

        # Hidden state initialization
        h_0 = torch.zeros(1, b_size, RoInflectModule._conf_lstm_size)
        h_0 = h_0.to(device=_device)
        # Internal state initialization
        c_0 = torch.zeros(1, b_size, RoInflectModule._conf_lstm_size)
        c_0 = c_0.to(device=_device)

        # Propagate input through LSTM, get output and state information
        lstm_outputs, (h_n, c_n) = self._layer_lstm(
            packed_embeddings, (h_0, c_0))
        out = self._layer_linear_1(h_n)
        out = self._tanh(out)
        out = self._layer_drop(out)
        out = self._layer_linear_2(out)
        out = self._sigmoid(out)

        # Remove the (1, ...) from the shape as it is 1
        # for left-to-right LSTM
        return out.view(b_size, -1)


class RoInflectDataset(Dataset):
    """Implements a PyTorch dataset over words
    and possible MSDs NumPy tuples of arrays."""

    def __init__(self, dataset: list):
        super().__init__()
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


class RoInflect(object):
    """This class implements a RNN to recognize the mapping
    from the content (!) word form to the possible MSDs of it.
    That is, it learns ambiguity classes for nouns, verbs,
    adjectives and adverbs."""

    _conf_dev_size = 0.1

    def __init__(self, lexicon: Lex) -> None:
        self._lexicon = lexicon
        self._msd = self._lexicon.get_msd_object()
        # Use self._add_word_to_dataset() to update these
        self._dataset = {}
        self._charid = 1
        self._charmap = {'UNK': 0}
        self._cache = {}
        self.load_cache()

    def _add_word_to_dataset(self, word: str, msds: list):
        if word not in self._dataset:
            self._dataset[word] = []
        # end if

        for m in msds:
            if m not in self._dataset[word]:
                self._dataset[word].append(m)
            # end if
        # end for

        for c in word:
            if c not in self._charmap:
                self._charmap[c] = self._charid
                self._charid += 1
            # end if
        # end for

    def _build_io_vectors(self, word: str, msds: list) -> tuple:
        x = np.zeros(len(word), dtype=np.int32)

        for i in range(len(word)):
            c = word[i]

            if c in self._charmap:
                x[i] = self._charmap[c]
            # end if
        # end for

        y = np.zeros(self._msd.get_output_vector_size(), dtype=np.float32)

        for m in msds:
            if self._msd.is_valid_msd(m):
                y += self._msd.msd_reference_vector(m)
            # end if
        # end for

        return (x, y)

    def _do_one_epoch(self, epoch: int, dataloader: DataLoader):
        """Does one epoch of NN training."""
        running_loss = 0.
        epoch_loss = []
        counter = 0

        for inputs, target_labels in tqdm(dataloader, desc=f'Epoch {epoch}'):
            counter += 1

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self._model(x=inputs)

            # Compute the loss and its gradients
            loss = self._loss_fn(outputs, target_labels)
            loss.backward()

            # Adjust learning weights and learning rate schedule
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if counter % 500 == 0:
                # Average loss per batch
                average_running_loss = running_loss / 500
                print(f'\n  -> batch {counter}/{len(dataloader)} loss: {average_running_loss:.5f}',
                      file=sys.stderr, flush=True)
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print(
            f'  -> average epoch {epoch} loss: {average_epoch_loss:.5f}', file=sys.stderr, flush=True)

    def _test(self, dataloader: DataLoader):
        """Tests the model with the dev set."""

        correct = 0
        predicted = 0
        existing = 0

        for inputs, target_labels in tqdm(dataloader, desc=f'Eval'):
            outputs = self._model(x=inputs)
            predicted_labels = (outputs >= 0.5)
            target_labels = target_labels.to(dtype=torch.int32).to(dtype=torch.bool)
            correct_labels = torch.logical_and(predicted_labels, target_labels)
            predicted += torch.sum(predicted_labels).item()
            existing += torch.sum(target_labels).item()
            correct += torch.sum(correct_labels).item()
        # end for

        prec = correct / predicted
        rec = correct / existing
        f1 = 2 * prec * rec / (prec + rec)

        print(f'P(1) = {prec:.5f}', file=sys.stderr, flush=True)
        print(f'R(1) = {rec:.5f}', file=sys.stderr, flush=True)
        print(f'F1(1) = {f1:.5f}', file=sys.stderr, flush=True)

    def _roinfl_collate_fn(self, batch) -> tuple:
        batch_sequences = []
        batch_labels = []

        for x, y in batch:
            batch_sequences.append(torch.tensor(
                x, dtype=torch.long).to(device=_device))
            batch_labels.append(y)
        # end for

        y_array = np.array(batch_labels, dtype=np.float32)

        return \
            batch_sequences, \
            torch.tensor(y_array, dtype=torch.float32).to(device=_device)

    def train(self):
        # Read training data
        self._read_training_data()

        # Build model
        self._model = self._build_pt_model()
        self._loss_fn = nn.BCELoss()
        self._optimizer = RMSprop(self._model.parameters(), lr=1e-3)

        # Build NumPy dataset
        np_dataset_train = []
        word_list = list(self._dataset.keys())

        for i in range(len(word_list)):
            if (i + 1) % 100000 == 0:
                print(stack()[0][3] + f": computed {i + 1}/{len(word_list)} data samples",
                      file=sys.stderr, flush=True)
            # end if

            w = word_list[i]
            (x_w, y_w) = self._build_io_vectors(w, self._dataset[w])
            np_dataset_train.append((x_w, y_w))
        # end for dataset

        shuffle(np_dataset_train)
        np_dataset_dev = []
        devlen = int(len(np_dataset_train) * RoInflect._conf_dev_size)

        while len(np_dataset_dev) < devlen:
            np_dataset_dev.append(np_dataset_train.pop())
        # end while

        # Build PyTorch datasets
        pt_dataset_train = RoInflectDataset(dataset=np_dataset_train)
        pt_dataset_dev = RoInflectDataset(dataset=np_dataset_dev)

        train_dataloader = DataLoader(
            dataset=pt_dataset_train, batch_size=256, shuffle=True, collate_fn=self._roinfl_collate_fn)
        dev_dataloader = DataLoader(
            dataset=pt_dataset_dev, batch_size=64, shuffle=False, collate_fn=self._roinfl_collate_fn)

        for ep in range(1, 11):
            # Fit model for one epoch
            self._model.train(True)
            self._do_one_epoch(epoch=ep, dataloader=train_dataloader)

            # Test model
            # Put model into eval mode first
            self._model.eval()
            self._test(dataloader=dev_dataloader)
        # end for

        # Save model
        self._save_pt_model()

    def _build_pt_model(self) -> RoInflectModule:
        return RoInflectModule(
            char_vocab_dim=self._charid,
            msd_vector_dim=self._msd.get_output_vector_size()
        )

    def _save_pt_model(self):
        torchmodelfile = os.path.join(ROINFLECT_MODEL_FOLDER, 'model.pt')
        torch.save(self._model.state_dict(), torchmodelfile)
        self._save_char_map()

    def _save_char_map(self):
        print(stack()[0][3] + ": saving file {0}".format(ROINFLECT_CHARID_FILE),
              file=sys.stderr, flush=True)

        with open(ROINFLECT_CHARID_FILE, mode="w", encoding="utf-8") as f:
            # Write the next char ID which is also the length
            # of the char vocabulary
            print("{0!s}".format(self._charid), file=f, flush=True)

            # Write the Unicode properties dictionary
            for c in self._charmap:
                print("{0}\t{1!s}".format(
                    c, self._charmap[c]), file=f, flush=True)
            # end for
        # end with

    def _load_char_map(self):
        print(stack()[0][3] + ": loading file {0}".format(ROINFLECT_CHARID_FILE),
              file=sys.stderr, flush=True)

        first_line = True

        with open(ROINFLECT_CHARID_FILE, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()

                if first_line:
                    self._charid = int(line)
                    first_line = False
                else:
                    parts = line.split('\t')
                    self._charmap[parts[0]] = int(parts[1])
                # end if
            # end for
        # end with

    def load(self):
        self._load_char_map()
        self._model = self._build_pt_model()
        torchmodelfile = os.path.join(ROINFLECT_MODEL_FOLDER, 'model.pt')
        self._model.load_state_dict(torch.load(
            torchmodelfile, map_location=_device))
        # Put model into eval mode.
        # It is only used for inferencing.
        self._model.eval()

    def save_cache(self) -> None:
        with open(ROINFLECT_CACHE_FILE, mode='w', encoding='utf-8') as f:
            for word in sorted(self._cache.keys()):
                print('{0}\t{1}'.format(word, ', '.join([m for m in self._cache[word]])), file=f)
            # end all words
        # end with

    def load_cache(self):
        if os.path.exists(ROINFLECT_CACHE_FILE):
            with open(ROINFLECT_CACHE_FILE, mode='r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()
                    parts = line.split('\t')
                    word = parts[0]
                    msds = parts[1].split(', ')
                    self._cache[word] = msds
                # end for
            # end with
        # end if

    def msd_prob_for_word(self, word: str, msd: str) -> float:
        """Returns the learned probabilit for the given word/MSD combination."""

        word_vector = self._build_io_vectors(word, [])
        word_tensor, _ = self._roinfl_collate_fn([word_vector])
        y_pred = self._model(x=word_tensor)
        # Copy tensor on GPU to CPU first
        y_pred = y_pred.cpu()
        y_pred = y_pred.detach().numpy()
        msd_idx = self._msd.msd_to_idx(msd)

        return y_pred[0, msd_idx]

    def ambiguity_class(self, word: str, min_msds: int = 3) -> list:
        """Returns a list of possible MSDs for the given word.
        The list was learned from the training corpus and the lexicon.
        If no MSD is found at `prob_thr`, this is automatically decreased by 1%
        to try and find some MSDs. `prob_thr` is automatically decreased until at least
        `min_msds` have been found."""

        word_key = word + '@' + str(min_msds)

        if word_key in self._cache:
            return self._cache[word_key]
        # end if

        word_vector = self._build_io_vectors(word, [])
        word_tensor, _ = self._roinfl_collate_fn([word_vector])
        y_pred = self._model(x=word_tensor)
        # Copy tensor on GPU to CPU first
        y_pred = y_pred.cpu()
        y_pred = y_pred.detach().numpy()
        # Default for the Precition/Recall computation of a '1'
        prob_thr = 0.5
        # Indexes of the positions in y_pred that are > prob_thr
        y_idx = np.nonzero(y_pred > prob_thr)[1]
        
        while y_idx.shape[0] < min_msds and prob_thr > 0.01:
            prob_thr -= 0.01
            y_idx = np.nonzero(y_pred > prob_thr)[1]
        # end if

        word_amb_class = []

        for i in y_idx:
            # Probability of MSD @ i is in y_pred[0, i]
            word_amb_class.append(self._msd.idx_to_msd(i))
        # end for

        if word_amb_class:
            print_error("unknown word '{0}' has ambiguity class [{1}]".format(
                word, ', '.join([x for x in word_amb_class])), stack()[0][3])

            self._cache[word_key] = word_amb_class
        # end if

        return word_amb_class

    def _read_training_data(self):
        print(stack()[0][3] + ": reading training file {0!s}".format(
            TBL_WORDFORM_FILE), file=sys.stderr, flush=True)

        with open(TBL_WORDFORM_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line.startswith('#'):
                    continue
                else:
                    parts = line.split()

                    if len(parts) == 3:
                        word = parts[0]
                        msd = parts[2]

                        if Lex.content_word_pos_pattern.match(msd):
                            word = Lex.repl_sgml_wih_utf8(word)
                            self._add_word_to_dataset(word, [msd])
                        # end if
                    # end if
                # end if
            # end all lines
        # end with

        rrt_training_file = os.path.join(
            "data", "training", "tagger", "ro_rrt-ud-train.tab")

        print(stack()[0][3] + ": reading training file {0!s}".format(
            rrt_training_file), file=sys.stderr, flush=True)

        with open(rrt_training_file, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if line:
                    parts = line.split()

                    if len(parts) == 3:
                        word = parts[0]
                        msd = parts[2]

                        if Lex.content_word_pos_pattern.match(msd):
                            self._add_word_to_dataset(word, [msd])
                        # end if
                    # end if
                # end if
            # end all lines
        # end while

if __name__ == '__main__':
    lexi = Lex()
    morpho = RoInflect(lexi)
    morpho.train()
