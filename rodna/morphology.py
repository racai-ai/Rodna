import sys
import os
import numpy as np
import tensorflow as tf
from random import shuffle
from inspect import stack

from utils.errors import print_error
from utils.Lex import Lex
from config import TBL_WORDFORM_FILE, \
    ROINFLECT_MODEL_FOLDER, ROINFLECT_CHARID_FILE, \
    ROINFLECT_CACHE_FILE


class RoInflect(object):
    """This class implements a RNN to recognize the mapping
    from the word form to the possible MSDs of it."""

    _conf_keep_msd_prob_threshold = 0.01
    _conf_dev_size = 0.1
    _conf_char_embed_size = 32
    _conf_lstm_size = 256
    _conf_dense_size = 512

    def __init__(self, lexicon: Lex) -> None:
        self._lexicon = lexicon
        self._msd = self._lexicon.get_msd_object()
        self._M = self._lexicon.longestwordlen
        # Use self._add_word_to_dataset() to update these
        self._dataset = {}
        self._charid = 2
        self._charmap = {'UNK': 0, ' ': 1}
        self._cache = {}
        self.load_cache()

    def _add_word_to_dataset(self, word: str, msds: list) -> None:
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
        if len(word) > self._M:
            word = word[-self._M:]
        else:
            while len(word) < self._M:
                word = ' ' + word
            # end while
        # end if

        x = np.zeros(self._M, dtype=np.int32)

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

    def train(self) -> None:
        # Read training data
        self._read_training_data()

        # Build model
        self._model = self._build_keras_model()
        self._model.summary()

        # Compile model
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        self._model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=[
                tf.keras.metrics.TruePositives(),
                # Uncomment if needed.
                # tf.keras.metrics.TrueNegatives(),
                # tf.keras.metrics.FalsePositives(),
                # tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

        # Build data tensor
        m = len(self._dataset)
        x_input = np.empty((m, self._M), dtype=np.int32)
        y_label = np.empty(
            (m, self._msd.get_output_vector_size()), dtype=np.float32)
        word_list = list(self._dataset.keys())

        shuffle(word_list)

        for i in range(len(word_list)):
            if i > 0 and i % 100000 == 0:
                print(stack()[0][3] + ": computed {0!s}/{1!s} data samples".format(i, m),
                      file=sys.stderr, flush=True)
            # end if

            w = word_list[i]
            (x_w, y_w) = self._build_io_vectors(w, self._dataset[w])
            x_input[i, :] = x_w
            y_label[i, :] = y_w
        # end for dataset

        # Fit model
        self._model.fit(x=x_input, y=y_label, epochs=50, batch_size=256,
                        validation_split=RoInflect._conf_dev_size)
        # Save model
        self._save_keras_model()        

    def _build_keras_model(self):
        x = tf.keras.layers.Input(
            shape=(self._M,), dtype='int32', name="char-id-input")
        e = tf.keras.layers.Embedding(
            self._charid, RoInflect._conf_char_embed_size, input_length=self._M)(x)
        l = tf.keras.layers.LSTM(
            RoInflect._conf_lstm_size, return_sequences=False)(e)
        d = tf.keras.layers.Dense(
            RoInflect._conf_dense_size, activation='tanh')(l)
        y = tf.keras.layers.Dense(self._msd.get_output_vector_size(
        ), activation='sigmoid', name="possible-msds")(d)

        return tf.keras.Model(inputs=x, outputs=y)

    def _save_keras_model(self):
        self._model.save(ROINFLECT_MODEL_FOLDER, overwrite=True)
        self._save_char_map()

    def _save_char_map(self) -> None:
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

    def _load_char_map(self) -> None:
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
        self._model = tf.keras.models.load_model(ROINFLECT_MODEL_FOLDER)
        print(stack()[0][3] + ": loading file {0}".format(ROINFLECT_CHARID_FILE),
              file=sys.stderr, flush=True)
        self._load_char_map()

    def save_cache(self) -> None:
        with open(ROINFLECT_CACHE_FILE, mode='w', encoding='utf-8') as f:
            for word in sorted(self._cache.keys()):
                print('{0}\t{1}'.format(word, ', '.join([m for m in self._cache[word]])), file=f)
            # end all words
        # end with

    def load_cache(self) -> None:
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

        (x_word, _) = self._build_io_vectors(word, [])
        x_word = np.reshape(x_word, (1, x_word.shape[0]))
        y_pred = self._model.predict(x=x_word)
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

    def _read_training_data(self) -> None:
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
    lexicon = Lex()
    morpho = RoInflect(lexicon)
    morpho.train()
