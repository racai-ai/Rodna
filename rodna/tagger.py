import sys
import os
import sys
import numpy as np
import tensorflow as tf
from inspect import stack

from utils.MSD import MSD
from utils.datafile import read_all_ext_files_from_dir
from rodna.splitter import RoSentenceSplitter
from rodna.tokenizer import RoTokenizer
from utils.Lex import Lex
from config import PREDICTED_AMB_CLASSES_FILE

_predict_str_const = ": predicted MSD {0}/{1:.5f} preferred over lexicon MSD {2}/{3:.5f} for word '{4}'"


class AccCallback(tf.keras.callbacks.Callback):
    """Accuracy callback for model.fit."""

    def __init__(self, ropt, gold, epochs):
        super().__init__()
        self._ropt = ropt
        self._goldSentences = gold
        self._epochs = epochs

    def _sent_to_str(self, sentence):
        str_toks = [x[0] + "/" + x[1] + "/" + ",".join(x[2]) for x in sentence]
        return " ".join(str_toks)

    def compute_metric(self, epoch):
        total_words = 0
        error_words = 0
        f = None
        
        if epoch + 1 == self._epochs:
            f = open("epoch-" + str(epoch + 1) +
                     "-debug.txt", mode='w', encoding='utf-8')
        # end if

        for sentence in self._goldSentences:
            pt_sentence = self._ropt._run_sentence(sentence)

            if len(sentence) != len(pt_sentence):
                print(stack()[0][3] + ": sentences differ in size!",
                        file=sys.stderr, flush=True)
                continue
            # end if

            total_words += len(sentence)

            for i in range(len(sentence)):
                word = sentence[i][0]
                gold_msd = sentence[i][1]
                pred_msd = pt_sentence[i][1]

                if gold_msd != pred_msd:
                    error_words += 1
                # end if

                if f is not None:
                    is_train_word = False
                    is_lex_word = False
                    is_wemb_word = False

                    if word.lower() in self._ropt._train_words:
                        is_train_word = True
                    # end if

                    if self._ropt._lexicon.is_lex_word(word):
                        is_lex_word = True
                    # end if

                    if self._ropt._lexicon.is_wemb_word(word):
                        is_wemb_word = True
                    # end if

                    if gold_msd != pred_msd:
                        print("{0} {1} {2} T:{3!s} L:{4!s} E:{5!s} {6}".format(word, gold_msd, pred_msd,
                                                                            is_train_word, is_lex_word, is_wemb_word, self._sent_to_str(sentence)), file=f, flush=True)
                    # end if
                # end if f is set
            # end all words in sentence
        # end for all gold sentences

        if f is not None:
            f.close()
        # end if

        return (total_words, error_words)

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % 5 == 0:
            (total, errors) = self.compute_metric(epoch)
            acc = (total - errors) / total
            logs["acc"] = acc

            print(stack()[0][3] + ": dev accuracy at epoch {0!s} is Acc = {1:.5f}".format(
                epoch + 1, acc), file=sys.stderr, flush=True)
        # end if, expensive computation


class RoPOSTagger(object):
    """This class will do MSD POS tagging for Romanian.
    It will train/test the DNN models and also, given a string of Romanian text,
    it will split it in sentences, POS tag each sentence and return the list."""

    # How many words in a window to consider when constructing a sample.
    # This is the Tx value in the Deep Learning course.
    _conf_max_seq_length = 20
    # How much (%) to retain from the train data as dev/test sets
    _conf_dev_percent = 0.1
    # No test, for now, look at values on dev
    _conf_test_percent = 0.0
    # LSTM state size
    _conf_lstm_size_1 = 128
    _conf_lstm_size_2 = 128
    # Dense size to the classification layer
    _conf_map_hidden_size = 512
    # CharWNN configuration parameters
    _conf_charwnn_wemb_output = 32
    _conf_charwnn_k = 5
    _conf_charwnn_conv_output = 16
    _conf_epochs = 20
    # Special POS tagging features for Romanian
    # Computed here for any Romanian sentence
    # Format: str: int (which is the feature index)
    _romanian_pos_tagging_features = {
        # Word is at the beginning of the sentence
        "WORD_AT_BOS": 0,
        # Word is at the end of the sentence
        "WORD_AT_EOS": 1,
        # Word is some form of 'a fi'
        "TO_BE_AUX": 2,
        "TO_BE_MAIN": 3,
        # Resolve Afp/Rgp ambiguity
        "SHOULD_BE_ADVERB": 4,
        "SHOULD_BE_ADJECTIVE": 5
    }

    def __init__(self, splitter: RoSentenceSplitter):
        """Takes a trained instance of the RoSentenceSplitter object."""
        self._splitter = splitter
        self._tokenizer = splitter.get_tokenizer()
        self._uniprops = splitter.get_unicode_props()
        self._lexicon = splitter.get_lexicon()
        self._msd = self._lexicon.get_msd_object()
        self._predambclasses = self._read_pred_amb_classes(PREDICTED_AMB_CLASSES_FILE)
        # The fixed length of a word
        self._M = self._lexicon.longestwordlen
        self._charid = 4
        # The < and > are word boundaries. Spaces are used
        # to pad the word to length M
        self._charmap = {'UNK': 0, '<': 1, '>': 2, ' ': 3}

    def train(self, data_sentences: list = [], train_sentences: list = [], dev_sentences: list = [], test_sentences: list = []):
        # Either data_sentences is not None or
        # train_sentences, dev_sentences and test_sentences are not none
        if data_sentences and \
            not train_sentences and not dev_sentences and not test_sentences:
            # 1. Get the full word sequence from the train folder
            print(stack()[0][3] + ": got {0!s} sentences.".format(
                len(data_sentences)), file=sys.stderr, flush=True)

            # 2. Cut the full word sequence into maxSeqLength smaller sequences
            # and assign those randomly to train/dev/test sets
            print(stack()[0][3] + ": building train/dev/test samples",
                file=sys.stderr, flush=True)

            (train_sentences, train_examples, dev_sentences, dev_examples, test_sentences, test_examples) = self._build_train_samples(data_sentences)
        elif not data_sentences and \
            train_sentences and dev_sentences and test_sentences:
            (train_examples, dev_examples, test_examples) = self._build_train_samples_already_split(
                train_sentences, dev_sentences, test_sentences)
        # end if

        print(stack()[0][3] + ": got train set with {0!s} samples".format(
            len(train_examples)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got dev set with {0!s} samples".format(
            len(dev_examples)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got dev set with {0!s} sentences".format(
            len(dev_sentences)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got test set with {0!s} samples".format(
            len(test_examples)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got test set with {0!s} sentences".format(
            len(test_sentences)), file=sys.stderr, flush=True)

        # 3. Build the Unicode properties on the train set
        print(stack()[0][3] + ": building Unicode properties on train set",
              file=sys.stderr, flush=True)
        self._build_unicode_props(train_examples)

        # 3.1 Build the character map on the train set
        print(stack()[0][3] + ": building the character map on train set",
              file=sys.stderr, flush=True)
        self._add_sentences_to_map(train_sentences)

        # 4. Get train/dev/test numpy tensors
        def _generate_tensors(ml_type: str, examples: list) -> tuple:
            print(stack()[0][3] + ": building ENC/CLS {0} tensors".format(ml_type),
                  file=sys.stderr, flush=True)

            (x, xc, y_enc, y_cls) = self._build_model_io_tensors(examples)

            print(stack()[0][3] + ": x.shape is {0!s}".format(
                x.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": xc.shape is {0!s}".format(
                xc.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": ENC y.shape is {0!s}".format(
                y_enc.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": CLS y.shape is {0!s}".format(
                y_cls.shape), file=sys.stderr, flush=True)

            return (x, xc, y_enc, y_cls)
        # end def

        (x_train, xc_train, y_train_enc, y_train_cls) = _generate_tensors(
            'train', train_examples)
        (x_dev, xc_dev, y_dev_enc, y_dev_cls) = _generate_tensors('dev', dev_examples)

        input_dim = x_train.shape[2]
        encoding_dim = y_train_enc.shape[2]
        output_dim = y_train_cls.shape[2]

        # 5. Creating the tagging Keras model
        self._model = self._build_keras_model(
            input_dim, encoding_dim, output_dim,
            self._charid, self._M,
            RoPOSTagger._conf_charwnn_wemb_output, RoPOSTagger._conf_charwnn_k, RoPOSTagger._conf_charwnn_conv_output)

        # 6. Print model summary
        self._model.summary()

        # 6. Train the tagging model
        self._train_keras_model(train=(x_train, xc_train, y_train_enc, y_train_cls), dev=(
            x_dev, xc_dev, y_dev_enc, y_dev_cls), gold_sentences=dev_sentences)

    def _read_pred_amb_classes(self, file: str) -> list:
        aclasses = []

        with open(file, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split()

                if line.startswith('#'):
                    continue
                # skip comments

                aclasses.append((parts))
            # end for
        # end with

        return aclasses

    def _prefer_msd(self, word: str, pred_msd: str, pmp: float, lex_msd: str, lmp: float) -> tuple:
        """Chooses between the predicted MSD and the lexicon MSD for the given word.
        Goes through the options where the predicted MSD is better.
        Returns the lexicon MSD if no option is valid."""

        # 1. If one of the MSDs is included in the other one,
        # return the most informative one.
        if pred_msd.startswith(lex_msd):
            rp = pmp + lmp

            if rp > 1.0:
                rp = 1.0
            # end if

            print(stack()[0][3] + _predict_str_const.format(pred_msd,
                                                            rp, lex_msd, lmp, word), file=sys.stderr, flush=True)
            return (pred_msd, rp)
        # end if

        if lex_msd.startswith(pred_msd):
            rp = pmp + lmp

            if rp > 1.0:
                rp = 1.0
            # end if

            print(stack()[0][3] + ": lexicon MSD {0}/{1:.5f} preferred over predicted MSD {2}/{3:.5f} for word '{4}'".format(
                lex_msd, rp, pred_msd, pmp, word), file=sys.stderr, flush=True)
            return (lex_msd, rp)
        # end if

        # 2. If predicted is 'Np' and word is sentence-cased, leave Np.
        if pred_msd.startswith('Np') and Lex.sentence_case_pattern.match(word) and \
                not Lex.upper_case_pattern.match(word):
            return (pred_msd, pmp)
        # end if

        # 3. Consult the predicted ambiguity classes that could extend the lexicon.
        for (m1, m2) in self._predambclasses:
            if (pred_msd.startswith(m1) and lex_msd.startswith(m2)) or \
                    (pred_msd.startswith(m2) and lex_msd.startswith(m1)):
                print(stack()[0][3] + _predict_str_const.format(pred_msd,
                                                                pmp, lex_msd, lmp, word), file=sys.stderr, flush=True)
                return (pred_msd, pmp)
            # end if
        # end for

        # Default
        print(stack()[0][3] + ": default lexicon MSD {0}/{1:.5f} preferred over predicted MSD {2}/{3:.5f} for word '{4}'".format(
            lex_msd, lmp, pred_msd, pmp, word), file=sys.stderr, flush=True)
        return (lex_msd, lmp)

    def _most_prob_msd(self, word: str, y_pred: np.ndarray) -> tuple:
        y_best_idx = np.argmax(y_pred)
        best_pred_msd = self._msd.idx_to_msd(y_best_idx)
        best_pred_msd_p = y_pred[y_best_idx]

        if self._lexicon.is_lex_word(word):
            word_msds = self._lexicon.get_word_msd_ambiguity_class(word)

            if best_pred_msd in word_msds:
                # 1. Predicted MSD is in the ambiguity class for word.
                # Ideal.
                return (best_pred_msd, best_pred_msd_p)
            # end if

            # 2. If not, just choose the lexicon MSD that has the same
            # POS with the predicted MSD. Less than ideal, but still OK.
            # Neural net has learned a new MSD for word which might be or
            # might be not good...
            best_lex_msd = ""
            best_lex_msd_p = 0.0

            for msd in word_msds:
                if MSD.get_msd_pos(msd) == MSD.get_msd_pos(best_pred_msd):
                    msd_idx = self._msd.msd_to_idx(msd)
                    msd_p = y_pred[msd_idx]

                    if msd_p > best_lex_msd_p:
                        best_lex_msd_p = msd_p
                        best_lex_msd = msd
                    # end if
                # end if msd in lexicon for word has same POS with best predicted MSD
            # end for

            if best_lex_msd:
                return self._prefer_msd(word, best_pred_msd,
                    best_pred_msd_p, best_lex_msd, best_lex_msd_p)
            # end if

            # 3. Oops, the neural net just learned a very different MSD for word.
            # This is usually wrong. Just choose the best lexicon MSD instead.
            for msd in word_msds:
                msd_idx = self._msd.msd_to_idx(msd)
                msd_p = y_pred[msd_idx]

                if msd_p > best_lex_msd_p:
                    best_lex_msd_p = msd_p
                    best_lex_msd = msd
                # end if
            # end for

            return self._prefer_msd(word, best_pred_msd,
                best_pred_msd_p, best_lex_msd, best_lex_msd_p)
        # end if

        return (best_pred_msd, best_pred_msd_p)

    def _compute_sentence_wide_features(self, sentence: list) -> None:
        """Will take a sentence and update it with sentence-wide POS tagging features."""

        for i in range(len(sentence)):
            parts = sentence[i]

            if i == 0:
                sentence[i] = (parts[0], parts[1], ["WORD_AT_BOS"])
            elif i == len(sentence) - 1:
                sentence[i] = (parts[0], parts[1], ["WORD_AT_EOS"])
            else:
                sentence[i] = (parts[0], parts[1], [])
            # end if

            if self._lexicon.is_to_be_word(parts[0]):
                win_size = 0
                is_aux = False

                for j in range(i + 1, len(sentence)):
                    if win_size == 3:
                        break

                    if self._lexicon.can_be_msd(sentence[j][0], "Vmp"):
                        sentence[i][2].append("TO_BE_AUX")
                        is_aux = True
                        break
                    # end if

                    win_size += 1
                # end for j

                if not is_aux:
                    sentence[i][2].append("TO_BE_MAIN")
                # end if

            if self._lexicon.has_ambiguity_class(parts[0], "Afpms-n", "Rgp") and not self._lexicon.can_be_msd(parts[0], "N"):
                if i < len(sentence) - 1 and \
                        (self._lexicon.can_be_msd(sentence[i + 1][0], "Af") or sentence[i + 1][0].lower() == 'de'):
                    sentence[i][2].append("SHOULD_BE_ADVERB")
                # end if

                if i >= 2 and \
                        ("TO_BE_MAIN" in sentence[i - 1][2] or "TO_BE_MAIN" in sentence[i - 2][2]):
                    sentence[i][2].append("SHOULD_BE_ADJECTIVE")
                # end if
            # end if
        # end for i

    def _run_sentence(self, sentence):
        tagged_sentence = []

        # 1. Build the fixed-length samples from the input sentence
        sent_samples = self._build_samples([sentence])
        # 2. Get the TAG tensors, only care about X
        (x_run, xc_run, _, _) = self._build_model_io_tensors(
            sent_samples, runtime=True)
        # 3. Use the TAG model to get MSD attribute vectors
        (_, y_pred_cls) = self._model.predict(x=[x_run, xc_run])

        for i in range(len(sent_samples)):
            sample = sent_samples[i]

            for j in range(len(sample)):
                if (i + j < len(sentence)):
                    if len(tagged_sentence) <= i + j:
                        tagged_sentence.append((sentence[i + j][0], []))
                    # end if

                    word = tagged_sentence[i + j][0]
                    y_pred = y_pred_cls[i, j, :]
                    (msd, msd_p) = self._most_prob_msd(word, y_pred)
                    ij_msd_best = tagged_sentence[i + j][1]

                    if ij_msd_best:
                        if msd_p > ij_msd_best[1]:
                            ij_msd_best[0] = msd
                            ij_msd_best[1] = msd_p
                        # end if
                    else:
                        ij_msd_best.append(msd)
                        ij_msd_best.append(msd_p)
                    # end if
                # end if
            # end for j
        # end for i

        tagged_sentence2 = []

        for parts in tagged_sentence:
            choice = parts[1]
            tagged_sentence2.append((parts[0], choice[0], choice[1]))
        # end for

        return tagged_sentence2

    def _train_keras_model(self, train: tuple, dev: tuple, gold_sentences: list):
        # Compile model
        adam_opt = tf.keras.optimizers.Adam(
            lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
        self._model.compile(
            loss={
                'encoding': 'binary_crossentropy',
                'classification': 'categorical_crossentropy'},
            optimizer=adam_opt,
            loss_weights={
                'encoding': 1.0,
                'classification': 1.0},
            metrics={
                'encoding': 'cosine_similarity',
                'classification': 'categorical_accuracy'}
        )

        x_train = train[0]
        xc_train = train[1]
        y_train = {
            'encoding': train[2],
            'classification': train[3]
        }
        x_dev = dev[0]
        xc_dev = dev[1]
        y_dev = {
            'encoding': dev[2],
            'classification': dev[3]
        }

        # Fit model
        acc_callback = AccCallback(
            self, gold_sentences, RoPOSTagger._conf_epochs)
        self._model.fit(x=[x_train, xc_train], y=y_train, epochs=RoPOSTagger._conf_epochs, batch_size=128,
                        shuffle=True, validation_data=([x_dev, xc_dev], y_dev), callbacks=[acc_callback])

    def _build_keras_model(self,
            input_vector_size: int,
            msd_encoding_vector_size: int,
            output_vector_size: int,
            # Size of the char-based one-hot vocabulary
            char_vocabulary_size: int,
            # Maximum size of the word in chars
            max_word_length: int,
            # Size of the embedding for a character
            d_chr: int,
            # Size of the moving window (affix)
            k_chr: int,
            # Number of output dimensions
            cl_u: int
            ) -> tf.keras.Model:
        # CharWNN neural net: http://proceedings.mlr.press/v32/santos14.pdf
        # This neural net models the morphology of the word, i.e. inflectional/derivational affixes.
        xc = tf.keras.layers.Input(
            shape=(RoPOSTagger._conf_max_seq_length, max_word_length), dtype='int32')
        ch_wemb = tf.keras.layers.Embedding(
            char_vocabulary_size, d_chr, input_length=max_word_length)(xc)
        conv_1d = tf.keras.layers.Conv1D(cl_u, k_chr, activation=None, use_bias=True)(ch_wemb)
        conv_1d_rsh = tf.keras.layers.Reshape((conv_1d.shape[1] * conv_1d.shape[2], conv_1d.shape[3]))(conv_1d)
        max_pool = tf.keras.layers.GlobalMaxPool1D(data_format='channels_first')(conv_1d_rsh)
        conv_1d_rsh2 = tf.keras.layers.Reshape((conv_1d.shape[1], conv_1d.shape[2]))(max_pool)
        # End CharWNN Keras implementation.

        x = tf.keras.layers.Input(shape=(RoPOSTagger._conf_max_seq_length,
                                         input_vector_size), dtype='float32')
        x_xc_conc = tf.keras.layers.Concatenate()([x, conv_1d_rsh2])
        bd_lstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(RoPOSTagger._conf_lstm_size_1, return_sequences=True))(x_xc_conc)
        drop_1 = tf.keras.layers.Dropout(0.25)(bd_lstm_1)
        msd_enc = tf.keras.layers.Dense(
            msd_encoding_vector_size, activation='sigmoid', name='encoding')(drop_1)
        
        # This is the simpler network
        #dense_1 = tf.keras.layers.Dense(RoPOSTagger._conf_map_hidden_size, activation='tanh')(msd_enc)
        #drop_2 = tf.keras.layers.Dropout(0.25)(dense_1)
        
        # This is the variant with the MSD language model
        bd_lsdm_2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(RoPOSTagger._conf_lstm_size_2, return_sequences=True))(msd_enc)
        dense_2 = tf.keras.layers.Dense(
            RoPOSTagger._conf_map_hidden_size, activation='tanh')(bd_lsdm_2)
        drop_2 = tf.keras.layers.Dropout(0.25)(dense_2)
        
        dense_3 = tf.keras.layers.Dense(
            output_vector_size)(drop_2)
        msd_cls = tf.keras.layers.Activation(
            'softmax', name='classification')(dense_3)

        return tf.keras.Model(inputs=[x, xc], outputs=[msd_enc, msd_cls])

    def _build_model_io_tensors(self, data_samples, runtime=False):
        # No of examples
        m = len(data_samples)
        # This should be equal to _conf_max_seq_length
        tx = len(data_samples[0])
        # assert Tx == _conf_max_seq_length
        # This is the size of the input vector
        n = -1
        x_tensor = None
        xc_tensor = np.empty((m, tx, self._M), dtype=np.int32)
        # Ys for the Keras MSD encoding part
        y_tensor_enc = None
        # Ys for the Keras MSD classification part
        y_tensor_cls = None

        # sample size is Tx, the number of time steps
        for i in range(len(data_samples)):
            sample = data_samples[i]
            # We should have that assert len(sample) == tx

            if i > 0 and i % 1000 == 0:
                print(stack()[0][3] + ": processed {0!s}/{1!s} samples".format(
                    i, len(data_samples)), file=sys.stderr, flush=True)
            # end if


            for j in range(len(sample)):
                parts = sample[j]
                word = parts[0]
                msd = parts[1]
                feats = parts[2]
                tlabel = self._tokenizer.tag_word(word)

                if not runtime:
                    y_in = self._msd.msd_input_vector(msd)
                    y_out = self._msd.msd_reference_vector(msd)
                else:
                    y_in = self._msd.get_x_input_vector()
                    y_out = self._msd.get_x_reference_vector()
                # end if

                label_features = self._tokenizer.get_label_features(tlabel)
                uni_features = self._uniprops.get_unicode_features(word)
                lexical_features = self._lexicon.get_word_features_for_pos_tagging(
                    word, self._msd, feats, RoPOSTagger._romanian_pos_tagging_features)

                # This is the featurized version of a word in the sequence
                x = np.concatenate(
                    (label_features, uni_features, lexical_features))

                if n == -1:
                    n = x.shape[0]
                    x_tensor = np.empty((m, tx, n), dtype=np.float32)
                    y_tensor_enc = np.empty(
                        (m, tx, self._msd.get_input_vector_size()), dtype=np.float32)
                    y_tensor_cls = np.empty(
                        (m, tx, self._msd.get_output_vector_size()), dtype=np.float32)
                #else:
                #    assert (n,) == x.shape

                # Computing xc for word
                xc = self._get_word_onehot_char_vector(word)

                x_tensor[i, j, :] = x
                xc_tensor[i, j, :] = xc
                y_tensor_enc[i, j, :] = y_in
                y_tensor_cls[i, j, :] = y_out
            # end j
        # end i

        return (x_tensor, xc_tensor, y_tensor_enc, y_tensor_cls)

    def _build_samples(self, sentences: list) -> list:
        all_samples = []

        # Assemble examples with required, fixed length
        for sentence in sentences:
            # Make all sentences have RoPOSTag.maxSeqLength at least.
            p_sentence = self._pad_sentence(sentence)

            for i in range(len(p_sentence)):
                left = i
                right = i + RoPOSTagger._conf_max_seq_length

                if right > len(p_sentence):
                    break
                # end if

                all_samples.append(p_sentence[left:right])
            # end for i
        # end all sentences

        return all_samples

    def _build_train_samples_already_split(self, train_sentences: list, dev_sentences: list, test_sentences: list):
        """Builds the from the predefined, offered split."""

        def count_words(sentences: list) -> int:
            word_count = 0

            for sentence in sentences:
                word_count += len(sentence)
            # end for

            return word_count
        # end def

        print(stack()[0][3] + ": there are {0!s} words in the train set".format(
            count_words(train_sentences)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": there are {0!s} words in the dev set".format(
            count_words(dev_sentences)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": there are {0!s} words in the test set".format(
            count_words(test_sentences)), file=sys.stderr, flush=True)

        self._train_words = set()

        # Add diagnostic info
        for sentence in train_sentences:
            for parts in sentence:
                word = parts[0].lower()

                if word not in self._train_words:
                    self._train_words.add(word)
                # end if
            # end for
        # end for

        train_samples = self._build_samples(train_sentences)
        dev_samples = self._build_samples(dev_sentences)
        test_samples = self._build_samples(test_sentences)

        return (train_samples, dev_samples, test_samples)

    def _build_train_samples(self, sentences: list):
        """Builds the ML sets by segmenting the data_sequence into fixed-length chunks."""

        total_word_count = 0

        for sentence in sentences:
            total_word_count += len(sentence)
        # end for

        np.random.shuffle(sentences)

        train_sentences = []
        dev_sentences = []
        test_sentences = []
        dev_scount = int(
            RoPOSTagger._conf_dev_percent * total_word_count)
        test_scount = int(
            RoPOSTagger._conf_test_percent * total_word_count)

        print(stack()[0][3] + ": there are {0!s} words in the training data".format(
            total_word_count), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": there will be {0!s} words in the dev set".format(
            dev_scount), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": there will be {0!s} words in the test set".format(
            test_scount), file=sys.stderr, flush=True)

        current_word_count = 0

        while current_word_count < dev_scount:
            dev_sentences.append(sentences.pop(0))
            current_word_count += len(dev_sentences[-1])
        # end while

        print(stack()[0][3] + ": there are {0!s} words in the dev set".format(
            current_word_count), file=sys.stderr, flush=True)
        current_word_count = 0

        while current_word_count < test_scount:
            test_sentences.append(sentences.pop(0))
            current_word_count += len(test_sentences[-1])
        # end while

        print(stack()[0][3] + ": there are {0!s} words in the test set".format(
            current_word_count), file=sys.stderr, flush=True)
        train_sentences.extend(sentences)
        self._train_words = set()

        # Add diagnostic info
        for sentence in train_sentences:
            for parts in sentence:
                word = parts[0].lower()

                if word not in self._train_words:
                    self._train_words.add(word)
                # end if
            # end for
        # end for

        train_samples = self._build_samples(train_sentences)
        dev_samples = self._build_samples(dev_sentences)
        test_samples = self._build_samples(test_sentences)

        train_sentences.extend(sentences)
        return (train_sentences, train_samples, dev_sentences, dev_samples, test_sentences, test_samples)

    def _build_unicode_props(self, data_samples: list) -> None:
        for sample in data_samples:
            for parts in sample:
                word = parts[0]
                self._uniprops.add_unicode_props(word)
            # end for
        # end for

    def _pad_sentence(self, sentence: list) -> list:
        """If the sentence has less tokens than the trained
        seqence length, pad it with the last word, at the right,
        to reach RoPOSTag.maxSeqLength."""
        padded_sentence = sentence[:]

        while len(padded_sentence) < RoPOSTagger._conf_max_seq_length:
            padded_sentence.append(padded_sentence[-1])
        # end with

        return padded_sentence

    def read_tagged_file(self, file: str) -> list:
        """Will read in file and return a sequence of tokens from it
        each token with its assigned MSD."""
        sentences = []
        current_sentence = []
        line_count = 0

        with open(file, mode='r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                line = line.strip()
                parts = line.split()

                if not line and current_sentence:
                    self._compute_sentence_wide_features(current_sentence)
                    sentences.append(current_sentence)
                    current_sentence = []
                    continue
                # end if

                if len(parts) != 3:
                    print(stack()[0][3] + ": line {0!s} in file {1!s} is not well-formed!".format(
                        line_count, file), file=sys.stderr, flush=True)
                else:
                    current_sentence.append((parts[0], parts[2]))
                # end if
            # end all lines
        # end with

        return sentences

    ##### CharWNN methods
    def _add_word_to_map(self, word: str) -> None:
        for c in word:
            c = c.lower()

            if c not in self._charmap:
                self._charmap[c] = self._charid
                self._charid += 1
            # end if
        # end for

    def _add_sentences_to_map(self, sentences: list) -> None:
        """Sentences from the RoPOSTagger training set."""

        for sentence in sentences:
            for t in sentence:
                # Assumes t[0] is the word, t[1] is the MSD
                self._add_word_to_map(t[0])
            # end all tokens
        # end all sentences

    def _get_word_onehot_char_vector(self, word: str) -> np.ndarray:
        if len(word) > self._M:
            print(stack()[0][3] + \
                ": word '{0}' has length {1!s} > max. length of {2!s}".format(word, len(word), self._M),
                file=sys.stderr, flush=True)
            word = word[-(self._M - 2):]
        # end if

        # Add word boundaries
        word = '<' + word + '>'
        space_add_flag = True

        # Pad word with spaces
        while len(word) < self._M:
            if space_add_flag:
                word = ' ' + word
                space_add_flag = False
            else:
                word = word + ' '
                space_add_flag = True
            # end if
        # end while

        onhtsq = np.zeros(self._M, dtype=np.int32)

        for i in range(self._M):
            c = word[i].lower()

            if c in self._charmap:
                chi = self._charmap[c]
                onhtsq[i] = chi
            # end if
        # end for

        return onhtsq

if __name__ == '__main__':
    # Use this module to train the sentence splitter.
    tk = RoTokenizer()
    ss = RoSentenceSplitter(tk)
    ss.load_keras_model()
    tg = RoPOSTagger(ss)

    # When we have multiple files
    #data_sentences = []
    #tab_files = read_all_ext_files_from_dir(os.path.join(
    #    "data", "training", "tagger"), extension='.tab')
    #
    #for file in tab_files:
    #    print(stack()[
    #        0][3] + ": reading training file {0!s}".format(file), file=sys.stderr, flush=True)
    #    sentences = tg.read_tagged_file(file)
    #    data_sentences.extend(sentences)
    # end for
    #
    #tg.train(data_sentences)

    # For a given split, like in RRT
    training_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-train.tab")
    print(stack()[0][3] + ": reading training file {0!s}".format(
        training_file), file=sys.stderr, flush=True)
    training = tg.read_tagged_file(training_file)
    
    development_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-dev.tab")
    print(stack()[0][3] + ": reading development file {0!s}".format(
        development_file), file=sys.stderr, flush=True)
    development = tg.read_tagged_file(development_file)

    testing_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-test.tab")
    print(stack()[0][3] + ": reading testing file {0!s}".format(
        testing_file), file=sys.stderr, flush=True)
    testing = tg.read_tagged_file(testing_file)

    tg.train(train_sentences=training, dev_sentences=development, test_sentences=testing)
