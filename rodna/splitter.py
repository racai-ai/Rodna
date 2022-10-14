from typing import List, Tuple
import sys
import os
from inspect import stack
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch import Tensor
from tqdm import tqdm
from utils.CharUni import CharUni
from utils.Lex import Lex
from rodna.tokenizer import RoTokenizer
from utils.datafile import read_all_ext_files_from_dir, tok_file_to_tokens
from config import SENT_SPLITTER_MODEL_FOLDER, \
    SPLITTER_UNICODE_PROPERTY_FILE, SPLITTER_FEAT_LEN_FILE
from . import _device


class RoSentenceSplitterModule(nn.Module):
    """The PyTorch neural network module for sentence splitting."""

    # LSTM state size
    _conf_lstm_size = 64

    def __init__(self, feat_dim: int):
        super().__init__()
        self._layer_lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=RoSentenceSplitterModule._conf_lstm_size,
            batch_first=True,
            bidirectional=True)
        # We are going to concatenate the left-to-right and the
        # right-to-left LSTM output states.
        self._layer_linear = nn.Linear(
            in_features=2 * RoSentenceSplitterModule._conf_lstm_size,
            out_features=2
        )
        self._layer_drop = nn.Dropout(p=0.3)
        self._layer_logsoftmax = nn.LogSoftmax(dim=2)
        self.to(device=_device)

    def forward(self, x):
        b_size = x.shape[0]
        # Hidden state initialization
        h_0 = torch.zeros(
            2, b_size,
            RoSentenceSplitterModule._conf_lstm_size).to(device=_device)
        # Internal state initialization
        c_0 = torch.zeros(
            2, b_size,
            RoSentenceSplitterModule._conf_lstm_size).to(device=_device)

        # Propagate input through LSTM, get output and state information
        lstm_outputs, (h_n, c_n) = self._layer_lstm(x, (h_0, c_0))
        # Shape is (N, L, 2 * _conf_lstm_size)
        out = self._layer_drop(lstm_outputs)
        out = self._layer_linear(out)
        out = self._layer_logsoftmax(out)

        return out

class SSDataset(Dataset):
    """This is a sentence splitter dataset."""

    def __init__(self, samples: List[Tuple]):
        super().__init__()
        self._data_samples = samples

    def __len__(self) -> int:
        return len(self._data_samples)

    def __getitem__(self, index):
        return self._data_samples[index]


class RoSentenceSplitter(object):
    """This class will do sentence splitting for Romanian.
    It will train/test the DNN models and also, given a string of Romanian text,
    it will split it in sentences and return the list."""

    # Only two classes: [1, 0] for no sentence end and [0, 1] for sentence end.
    eos_label = "SENTEND"
    # If the average of the eosLabel class probabilities is larger than this,
    # we assign the eosLabel
    _conf_eos_prob = 0.5
    # How many words in a window to consider when constructing a sample.
    _conf_max_seq_length = 50
    # When we do sentence splitting, how many samples to run
    # through the NN at once.
    _conf_run_batch_length = 4096
    # How much (%) to retain from the train data as dev/test sets
    _conf_dev_percent = 0.1
    _conf_test_percent = 0.1
    _conf_epochs = 3

    def __init__(self, lexicon: Lex, tokenizer: RoTokenizer):
        self._tokenizer = tokenizer
        self._uniprops = CharUni()
        self._lexicon = lexicon

    def train(self, word_sequence: list):
        """Takes a long word sequence (RoTokenizer tokenized text with 'SENTEND' annotations),
        trains and tests the sentence splitting model and saves it to the SENT_SPLITTER_MODEL_FOLDER folder."""

        # 1. Get the full word sequence from the train folder
        print(stack()[0][3] + ": got a sequence with {0!s} words.".format(
            len(word_sequence)), file=sys.stderr, flush=True)

        # 2. Cut the full word sequence into maxSeqLength smaller sequences
        # and assign those randomly to train/dev/test sets
        print(stack()[0][3] + ": building train/dev/test samples",
              file=sys.stderr, flush=True)
        (train_examples, dev_examples,
         test_examples) = self._build_samples(word_sequence)
        print(stack()[0][3] + ": got train set with {0!s} examples".format(
            len(train_examples)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got dev set with {0!s} examples".format(
            len(dev_examples)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got test set with {0!s} examples".format(
            len(test_examples)), file=sys.stderr, flush=True)

        # 3. Build the Unicode properties on the train set
        print(stack()[0][3] + ": building Unicode properties on train set",
              file=sys.stderr, flush=True)
        self._build_unicode_props(train_examples)

        x_check, y_check = self._build_single_sample_input(
            sample=train_examples[0])
        self._features_dim = x_check.shape[2]
        self._model = self._build_pt_model()
        
        # Save the model as a class attribute
        self._train_pt_model(train=train_examples,
                             dev=dev_examples, test=test_examples)
        self._save_pt_model()

    def _build_pt_model(self) -> RoSentenceSplitterModule:
        """Builds the PyTorch NN for the splitter."""

        return RoSentenceSplitterModule(feat_dim=self._features_dim)

    def _split_collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        x_tensor = []
        y_tensor = []

        for s in batch:
            x, y = self._build_single_sample_input(sample=s)
            x_tensor.append(torch.tensor(x))
            y_tensor.append(torch.tensor(y, dtype=torch.long))
        # end for
        
        x_tensor = torch.cat(x_tensor, dim=0).to(_device)
        y_tensor = torch.cat(y_tensor, dim=0).to(_device)

        return x_tensor, y_tensor

    def _train_pt_model(self, train: List[List[Tuple]], dev: List[List[Tuple]], test: List[List[Tuple]]) -> None:
        # PyTorch datasets
        pt_dataset_train = SSDataset(samples=train)
        train_dataloader = DataLoader(
            dataset=pt_dataset_train, batch_size=16, shuffle=True, collate_fn=self._split_collate_fn)
        pt_dataset_dev = SSDataset(samples=dev)
        dev_dataloader = DataLoader(
            dataset=pt_dataset_dev, batch_size=16, shuffle=False, collate_fn=self._split_collate_fn)
        pt_dataset_test = SSDataset(samples=test)
        test_dataloader = DataLoader(
            dataset=pt_dataset_test, batch_size=16, shuffle=False, collate_fn=self._split_collate_fn)
        
        self._loss_fn = nn.NLLLoss()
        self._optimizer = Adam(self._model.parameters(), lr=1e-3)

        for ep in range(RoSentenceSplitter._conf_epochs):
            # Fit model for one epoch
            self._model.train(True)
            self._do_one_epoch(epoch=ep + 1, dataloader=train_dataloader)

            # Test model
            # Put model into eval mode first
            self._model.eval()
            self._test(dataloader=dev_dataloader, ml_set='dev')
        # end for

        self._model.eval()
        self._test(dataloader=test_dataloader, ml_set='test')

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

            # Have to swap axes for NLLLoss function
            # Classes are on the second dimension, dim=1
            outputs = torch.swapaxes(outputs, 1, 2)
            loss = self._loss_fn(outputs, target_labels)
            loss.backward()

            # Adjust learning weights and learning rate schedule
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if counter % 200 == 0:
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

    def _compute_metric(self, x, y):
        y_pred = self._model(x)
        gold_argmax = np.argmax(y, axis=-1)
        pred_argmax = np.argmax(y_pred, axis=-1)
        diff = gold_argmax - pred_argmax
        fn = np.sum(diff == 1)
        fp = np.sum(diff == -1)
        match_sum = gold_argmax + pred_argmax
        tp = np.sum(match_sum == 2)
        prec = 0.0
        rec = 0.0
        fm = 0.0

        if tp + fp > 0:
            prec = float(tp) / float(tp + fp)
        # end if

        if tp + fn > 0:
            rec = float(tp) / float(tp + fn)
        # end if

        if prec > 0.0 or rec > 0.0:
            fm = 2 * prec * rec / (prec + rec)
        # end if

        prec = float(int(prec * 10000.0)) / 10000.0
        rec = float(int(rec * 10000.0)) / 10000.0
        fm = float(int(fm * 10000.0)) / 10000.0

        return (prec, rec, fm)

    def _test(self, dataloader: DataLoader, ml_set: str):
        """Tests the model with the dev/test sets."""

        correct = 0
        predicted = 0
        existing = 0

        for inputs, target_labels in tqdm(dataloader, desc=f'Eval'):
            outputs = self._model(x=inputs)
            outputs = torch.exp(outputs)
            predicted_labels = (outputs >= 0.5).to(dtype=torch.int64)[:, :, 1]
            target_labels = target_labels.to(dtype=torch.bool)
            predicted_labels = predicted_labels.to(dtype=torch.bool)
            correct_labels = torch.logical_and(predicted_labels, target_labels)
            predicted += torch.sum(predicted_labels).item()
            existing += torch.sum(target_labels).item()
            correct += torch.sum(correct_labels).item()
        # end for

        prec = correct / predicted
        rec = correct / existing
        f1 = 2 * prec * rec / (prec + rec)

        print(f'P(1) = {prec:.5f} on {ml_set}', file=sys.stderr, flush=True)
        print(f'R(1) = {rec:.5f} on {ml_set}', file=sys.stderr, flush=True)
        print(f'F1(1) = {f1:.5f} on {ml_set}', file=sys.stderr, flush=True)

    def load(self):
        with open(SPLITTER_FEAT_LEN_FILE, mode='r', encoding='utf-8') as f:
            self._features_dim = int(f.readline().strip())
        # end with

        self._uniprops.load_unicode_props(SPLITTER_UNICODE_PROPERTY_FILE)
        self._model = self._build_pt_model()
        torchmodelfile = os.path.join(SENT_SPLITTER_MODEL_FOLDER, 'model.pt')
        self._model.load_state_dict(torch.load(
            torchmodelfile, map_location=_device))
        # Put model into eval mode.
        # It is only used for inferencing.
        self._model.eval()

    def _save_pt_model(self):
        torchmodelfile = os.path.join(SENT_SPLITTER_MODEL_FOLDER, 'model.pt')
        torch.save(self._model.state_dict(), torchmodelfile)
        self._uniprops.save_unicode_props(SPLITTER_UNICODE_PROPERTY_FILE)

        with open(SPLITTER_FEAT_LEN_FILE, mode='w', encoding='utf-8') as f:
            print(f'{self._features_dim}', file=f)
        # end with

    def sentence_split(self, input_text: str) -> list:
        """Will run the sentence splitter model on the input_text and return
        the list of tokenized sentences."""

        word_sequence = self._tokenizer.tokenize(input_text)
        # Keeps the number of artificially inserted spaces
        # so that we can remove them at the end.
        tail_extra_count = 0

        while len(word_sequence) < RoSentenceSplitter._conf_max_seq_length:
            # We have too few tokens to do sentence splitting.
            # Pad the sequence with spaces.
            word_sequence.append((' ', 'SPACE'))
            tail_extra_count += 1
        # end while

        # Add a list of floats P(EOS) to each token in word_sequence initially empty.
        for i in range(len(word_sequence)):
            parts = word_sequence[i]
            word_sequence[i] = (parts[0], parts[1], [])
        # end for i

        batch_i = []
        batch_count = 0
        x_batch = None
        remaining_items_no = len(word_sequence) - \
            RoSentenceSplitter._conf_max_seq_length + 1
        m = RoSentenceSplitter._conf_run_batch_length

        if remaining_items_no < RoSentenceSplitter._conf_run_batch_length:
            m = remaining_items_no
        # end if

        # Now assign the SENTEND labels probabilities to every part in word_sequence
        for i in range(len(word_sequence)):
            left = i
            right = i + RoSentenceSplitter._conf_max_seq_length

            if right > len(word_sequence):
                break
            # end if

            # Run the batch through the NN if enough samples
            if len(batch_i) == RoSentenceSplitter._conf_run_batch_length:
                print(stack()[0][3] + ": running NN batch #{0!s}, size = {1!s}".format(
                    batch_count, x_batch.shape[0]), file=sys.stderr, flush=True)
                # Pass X_batch through the neural net
                pt_x_batch = torch.tensor(x_batch).to(_device)
                y_pred = self._model(pt_x_batch)
                y_pred = torch.exp(y_pred)
                y_pred = y_pred.cpu().detach().numpy()

                for bi in batch_i:
                    left_b = bi
                    right_b = bi + RoSentenceSplitter._conf_max_seq_length

                    # Add the probabilities to every word of the sample
                    for bj in range(left_b, right_b):
                        parts = word_sequence[bj]
                        # This is the probability of the label SENTEND
                        parts[2].append(
                            y_pred[bi - batch_count * RoSentenceSplitter._conf_run_batch_length][bj - left_b][1])
                    # end for
                # end for

                batch_i = []
                batch_count += 1
                x_batch = None
                remaining_items_no -= RoSentenceSplitter._conf_run_batch_length

                if remaining_items_no < RoSentenceSplitter._conf_run_batch_length:
                    m = remaining_items_no
                else:
                    m = RoSentenceSplitter._conf_run_batch_length
                # end if
            # end if

            # Add current sample to batch
            for j in range(left, right):
                parts = word_sequence[j]
                word = parts[0]
                tlabel = parts[1]

                label_features = self._tokenizer.get_label_features(tlabel)
                uni_features = self._uniprops.get_unicode_features(word)
                lexical_features = self._lexicon.get_word_features(word)

                # This is the featurized version of a word in the sequence
                x = np.concatenate(
                    (label_features, uni_features, lexical_features))

                if x_batch is None:
                    n = x.shape[0]
                    tx = RoSentenceSplitter._conf_max_seq_length
                    x_batch = np.empty((m, tx, n), dtype=np.float32)
                # end if

                x_batch[i - batch_count *
                        RoSentenceSplitter._conf_run_batch_length, j - left, :] = x
            # end for j

            batch_i.append(i)
        # end all words in sequence

        # Run last batch through the NN as well
        if batch_i:
            print(stack()[0][3] + ": running final NN batch #{0!s}, size = {1!s}".format(
                batch_count, x_batch.shape[0]), file=sys.stderr, flush=True)
            # Pass X_batch through the neural net
            pt_x_batch = torch.tensor(x_batch).to(_device)
            y_pred = self._model(pt_x_batch)
            y_pred = torch.exp(y_pred)
            y_pred = y_pred.cpu().detach().numpy()

            for bi in batch_i:
                left_b = bi
                right_b = bi + RoSentenceSplitter._conf_max_seq_length

                # Add the probabilities to every word of the sample
                for bj in range(left_b, right_b):
                    parts = word_sequence[bj]
                    # This is the probability of the label SENTEND
                    parts[2].append(
                        y_pred[bi - batch_count * RoSentenceSplitter._conf_run_batch_length][bj - left_b][1])
                # end for bj
            # end for bi
        # end if last batch

        # Remove extra added spaces to meet
        # RoSentenceSplitter._conf_max_seq_length constraint.
        while tail_extra_count > 0:
            word_sequence.pop()
            tail_extra_count -= 1
        # end while

        sentences = []
        current_sentence = []

        # Average the P(EOS) for each word and if it's > RoSentenceSplitter._conf_eos_prob, assign the label.
        for i in range(len(word_sequence)):
            parts = word_sequence[i]
            p_avg = 0.0

            if parts[2]:
                p_avg = np.average(np.asarray(parts[2]))
            # end if

            # Assign the label here
            if p_avg > RoSentenceSplitter._conf_eos_prob:
                current_sentence.append((parts[0], parts[1]))
                sentences.append(current_sentence)
                current_sentence = []
            else:
                current_sentence.append((parts[0], parts[1]))
            # end if
        # end for i

        # Append the last sentence.
        if current_sentence:
            sentences.append(current_sentence)
        # end if

        splitter_sentences = []

        # Glue ABBRs and split punctuation that is longer than 1 char
        for sent in sentences:
            glued_sent = self._tokenizer.glue_tokens(tokens=sent)
            splitter_sentences.append(glued_sent)
        # end for

        return splitter_sentences

    @staticmethod
    def is_eos_label(parts):
        """Tests to see if a tokenized unit has been labeled with the 'end of sentence label'."""

        return len(parts) == 3 and parts[2] == RoSentenceSplitter.eos_label

    def _build_unicode_props(self, data_samples):
        for sample in data_samples:
            for parts in sample:
                word = parts[0]
                self._uniprops.add_unicode_props(word)
            # end for
        # end all samples

    def _build_samples(self, data_sequence: List[Tuple]) -> Tuple[List, List, List]:
        """Builds the ML sets by segmenting the data_sequence into fixed-length chunks.
        Returns the classic ML triad, train, dev and test sets."""

        negative_samples = []
        positive_samples = []

        # 1. Assemble positive/negative examples
        for i in range(len(data_sequence)):
            left = i
            right = i + RoSentenceSplitter._conf_max_seq_length

            if right > len(data_sequence):
                break
            # end if

            sample = []
            sample_is_positive = False

            for j in range(left, right):
                parts = data_sequence[j]
                sample.append(parts)

                if not sample_is_positive and \
                        RoSentenceSplitter.is_eos_label(parts):
                    sample_is_positive = True
                # end if

            if sample_is_positive:
                positive_samples.append(sample)
            else:
                negative_samples.append(sample)
            # end if
        # end for i

        # 2. Sample train/dev/test sets
        all_samples = []
        all_samples.extend(positive_samples)
        all_samples.extend(negative_samples)

        np.random.shuffle(all_samples)

        train_samples = []
        dev_samples = []
        test_samples = []
        dev_scount = int(
            RoSentenceSplitter._conf_dev_percent * len(all_samples))
        test_scount = int(
            RoSentenceSplitter._conf_test_percent * len(all_samples))

        while len(dev_samples) < dev_scount:
            dev_samples.append(all_samples.pop(0))
        # end while

        while len(test_samples) < test_scount:
            test_samples.append(all_samples.pop(0))
        # end while

        train_samples.extend(all_samples)

        return (train_samples, dev_samples, test_samples)

    def _build_single_sample_input(self, sample: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        tx = len(sample)
        n = -1

        for j in range(len(sample)):
            parts = sample[j]
            word = parts[0]
            tlabel = parts[1]
            y = 0

            if RoSentenceSplitter.is_eos_label(parts):
                y = 1
            # end if

            label_features = self._tokenizer.get_label_features(tlabel)
            uni_features = self._uniprops.get_unicode_features(word)
            lexical_features = self._lexicon.get_word_features(word)

            # This is the featurized version of a word in the sequence
            x = np.concatenate(
                (label_features, uni_features, lexical_features))

            if n == -1:
                n = x.shape[0]
                X = np.empty((1, tx, n), dtype=np.float32)
                Y = np.empty((1, tx), dtype=np.int32)
            # end if

            X[0, j, :] = x
            Y[0, j] = y
        # end for j in a sample

        return X, Y


if __name__ == '__main__':
    # Use this module to train the sentence splitter.
    lx = Lex()
    tk = RoTokenizer(lx)
    ss = RoSentenceSplitter(lx, tk)

    # Using the .tok files for now.
    tok_files = read_all_ext_files_from_dir(os.path.join(
        'data', 'training', 'splitter'), extension='.tok')
    long_token_sequence = []

    for file in tok_files:
        print(stack()[
              0][3] + ": reading training file {0!s}".format(file), file=sys.stderr, flush=True)
        file_wsq = tok_file_to_tokens(file)
        long_token_sequence.extend(file_wsq)
    # end all files

    ss.train(long_token_sequence)
