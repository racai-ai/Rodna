import sys
import os
from inspect import stack
import numpy as np
import tensorflow as tf
from pathlib import Path
from utils.CharUni import CharUni
from rodna.tokenizer import RoTokenizer
from utils.datafile import read_all_ext_files_from_dir, tok_file_to_tokens
from config import SENT_SPLITTER_MODEL_FOLDER, UNICODE_PROPERTY_FILE


class PRFCallback(tf.keras.callbacks.Callback):
    """Precision/Recall/F-measure callback for model.fit() to compute other measures."""

    def __init__(self, x, y):
        super().__init__()
        self.x_data = x
        self.y_gold = y

    def computeMetric(self, x, y):
        y_pred = self.model.predict(x)
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

    def on_epoch_end(self, epoch, logs):
        (P, R, F) = self.computeMetric(self.x_data, self.y_gold)

        logs["prec"] = P
        logs["rec"] = R
        logs["fmeas"] = F

        print(stack()[0][3] + ": {0} dev precision at epoch {1!s} is P = {2!s}".format(
            RoSentenceSplitter.eos_label, epoch + 1, P), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": {0} dev recall at epoch {1!s} is R = {2!s}".format(
            RoSentenceSplitter.eos_label, epoch + 1, R), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": {0} dev f-measure at epoch {1!s} is F1 = {2!s}".format(
            RoSentenceSplitter.eos_label, epoch + 1, F), file=sys.stderr, flush=True)


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
    # This is the Tx value in the Deep Learning course.
    _conf_max_seq_length = 50
    # LSTM state size
    _conf_lstm_state_size = 64
    # When we do sentence splitting, how many samples to run
    # through the NN at once.
    _conf_run_batch_length = 4096
    # How much (%) to retain from the train data as dev/test sets
    _conf_dev_percent = 0.1
    _conf_test_percent = 0.1

    def __init__(self, tokenizer: RoTokenizer):
        self._tokenizer = tokenizer
        self._uniprops = CharUni()
        self._lexicon = tokenizer.get_lexicon()

    def get_tokenizer(self):
        return self._tokenizer

    def get_unicode_props(self):
        return self._uniprops

    def get_lexicon(self):
        return self._lexicon

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

        # 4. Get train/dev/test numpy tensors
        print(stack()[0][3] + ": building train tensor",
              file=sys.stderr, flush=True)
        (x_train, y_train) = self._build_keras_model_input(train_examples)
        print(stack()[
              0][3] + ": X_train.shape is {0!s}".format(x_train.shape), file=sys.stderr, flush=True)
        print(stack()[
              0][3] + ": Y_train.shape is {0!s}".format(y_train.shape), file=sys.stderr, flush=True)

        print(stack()[0][3] + ": building dev/test tensors",
              file=sys.stderr, flush=True)
        (x_dev, y_dev) = self._build_keras_model_input(dev_examples)
        (x_test, y_test) = self._build_keras_model_input(test_examples)
        print(stack()[
              0][3] + ": X_dev.shape is {0!s}".format(x_dev.shape), file=sys.stderr, flush=True)
        print(stack()[
              0][3] + ": Y_dev.shape is {0!s}".format(y_dev.shape), file=sys.stderr, flush=True)
        print(stack()[
              0][3] + ": X_test.shape is {0!s}".format(x_test.shape), file=sys.stderr, flush=True)
        print(stack()[
              0][3] + ": Y_test.shape is {0!s}".format(y_test.shape), file=sys.stderr, flush=True)

        # We leave out the number of examples in a batch
        input_shape = (x_train.shape[1], x_train.shape[2])
        output_shape = (y_train.shape[1], y_train.shape[2])
        print(stack()[
              0][3] + ": input shape is {0!s}".format(input_shape), file=sys.stderr, flush=True)
        print(stack()[
              0][3] + ": output shape is {0!s}".format(output_shape), file=sys.stderr, flush=True)
        self._model = self._build_keras_model(
            input_shape, output_shape[1], a_units=RoSentenceSplitter._conf_lstm_state_size)
        self._model.summary()

        # Save the model as a class attribute
        self._train_keras_model(train=(x_train, y_train), dev=(x_dev, y_dev), test=(x_test, y_test))
        self._save_keras_model()

    def _build_keras_model(self, in_shape: tuple, out_dim: int, a_units: int = 64) -> tf.keras.Model:
        """in_shape is of shape (maxSeqLength, len(features)) where 'features' is the input vector
        and out_dim is the length of the classification, output vector, e.g. 2 in this case."""

        # Ignore the batch_size or m, the number of training examples.
        # It is added by the Input layer.
        X = tf.keras.layers.Input(shape=in_shape, dtype='float32')
        # return_sequences = True tells the model to output a at each time step
        L = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            a_units, return_sequences=True))(X)
        # Connect a dense layer to a which has shape (a_units,)
        L = tf.keras.layers.Dense(out_dim)(L)
        # Activate it with the softmax function
        Y = tf.keras.layers.Activation('softmax')(L)

        return tf.keras.Model(X, Y)

    def _train_keras_model(self, train: tuple, dev: tuple, test: tuple) -> None:
        (x_train, y_train) = train
        (x_dev, y_dev) = dev
        (x_test, y_test) = test

        # Compile model
        prf_callback = PRFCallback(x_dev, y_dev)
        self._model.compile(loss='categorical_crossentropy',
                            optimizer='Adam', metrics=['categorical_accuracy'])

        # Fit model (train)
        # validation_data = dev as argument if you want
        self._model.fit(x_train, y_train, epochs=10,
                        batch_size=128, shuffle=True, callbacks=[prf_callback])

        (P, R, F) = prf_callback.computeMetric(x_test, y_test)
        print(stack()[0][3] + ": {0} test Precision is P = {1!s}".format(
            RoSentenceSplitter.eos_label, P), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": {0} test Recall is R = {1!s}".format(
            RoSentenceSplitter.eos_label, R), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": {0} test F-measure is F1 = {1!s}".format(
            RoSentenceSplitter.eos_label, F), file=sys.stderr, flush=True)

    def load(self):
        self._model = tf.keras.models.load_model(SENT_SPLITTER_MODEL_FOLDER)

        if Path(UNICODE_PROPERTY_FILE).is_file():
            print(stack()[0][3] + ": loading file {0}".format(
                UNICODE_PROPERTY_FILE), file=sys.stderr, flush=True)
            self._uniprops.load_unicode_props(UNICODE_PROPERTY_FILE)
        else:
            raise RuntimeError("File {0} was not found. You must train the sentence splitter before using it.".format(
                UNICODE_PROPERTY_FILE))

    def _save_keras_model(self):
        self._model.save(SENT_SPLITTER_MODEL_FOLDER, overwrite=True)
        self._uniprops.save_unicode_props(UNICODE_PROPERTY_FILE)

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
                y_pred = self._model.predict(x_batch)

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
            y_pred = self._model.predict(x_batch)

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

        return sentences

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

    def _build_samples(self, data_sequence: list) -> tuple:
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

                if not sample_is_positive and RoSentenceSplitter.is_eos_label(parts):
                    sample_is_positive = True

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

    def _build_keras_model_input(self, data_samples: list) -> tuple:
        # No of examples
        m = len(data_samples)
        # This should be equal to maxSeqLength
        tx = len(data_samples[0])
        # assert Tx == RoSentSplit.maxSeqLength
        # This is the size of the input vector
        n = -1
        X = None
        Y = None

        # sample size is Tx, the number of time steps
        for i in range(len(data_samples)):
            sample = data_samples[i]
            # We must have that len(sample) == tx

            for j in range(len(sample)):
                parts = sample[j]
                word = parts[0]
                tlabel = parts[1]
                y = np.zeros(2)

                if RoSentenceSplitter.is_eos_label(parts):
                    y[1] = 1.0
                else:
                    y[0] = 1.0

                label_features = self._tokenizer.get_label_features(tlabel)
                uni_features = self._uniprops.get_unicode_features(word)
                lexical_features = self._lexicon.get_word_features(word)

                # This is the featurized version of a word in the sequence
                x = np.concatenate(
                    (label_features, uni_features, lexical_features))

                if n == -1:
                    n = x.shape[0]
                    X = np.empty((m, tx, n), dtype=np.float32)
                    Y = np.empty((m, tx, 2), dtype=np.float32)
                #else:
                #    assert (n,) == x.shape

                X[i, j, :] = x
                Y[i, j, :] = y
            # end for j in a sample
        # end for i with all samples

        return (X, Y)


if __name__ == '__main__':
    # Use this module to train the sentence splitter.
    tk = RoTokenizer()
    ss = RoSentenceSplitter(tk)

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
