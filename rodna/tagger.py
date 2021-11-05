import sys
import os
import sys
from math import isclose
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.math_ops import _ReductionDims
import tensorflow_addons as tfa
from inspect import stack
from utils.CharUni import CharUni

from utils.MSD import MSD
from utils.datafile import read_all_ext_files_from_dir
from .splitter import RoSentenceSplitter
from .tokenizer import RoTokenizer
from .features import RoFeatures
from .morphology import RoInflect
from utils.Lex import Lex
from config import EMBEDDING_VOCABULARY_FILE, TAGGER_MODEL_FOLDER, \
    TAGGER_UNICODE_PROPERTY_FILE

_zero_word = '_ZERO_'
_unk_word = '_UNK_'

# Disable GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Enable 'as needed' GPU memory allocation
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class AccCallback(tf.keras.callbacks.Callback):
    """Accuracy callback for model.fit."""

    def __init__(self, ropt, gold, epochs, with_tt: bool, run_strat: str):
        super().__init__()
        self._ropt = ropt
        self._goldsentences = gold
        self._epochs = epochs
        self._withtt = with_tt
        self._runstrat = run_strat

    def _sent_to_str(self, sentence):
        str_toks = [x[0] + "/" + x[1] + "/" + ",".join(x[2]) for x in sentence]
        return " ".join(str_toks)

    def compute_metric(self, epoch):
        total_words = 0
        error_words = 0
        f = None
        ff = None

        if epoch + 1 == self._epochs:
            f = open("epoch-" + str(epoch + 1) +
                     "-debug.txt", mode='w', encoding='utf-8')
            ff = open("epoch-" + str(epoch + 1) +
                      "-debug2.txt", mode='w', encoding='utf-8')
        # end if

        for sentence in self._goldsentences:
            pt_sentence = self._ropt._run_sentence(sentence, self._withtt, self._runstrat, ff)

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
            ff.close()
        # end if

        return (total_words, error_words)

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % 5 == 0:
            (total, errors) = self.compute_metric(epoch)
            acc = (total - errors) / total
            logs["acc"] = acc

            print(stack()[0][3] + ": dev accuracy at epoch {0!s} is Acc = {1:.5f}".format(
                epoch + 1, acc), file=sys.stderr, flush=True)

            self._ropt._romorphology.save_cache()
        # end if, expensive computation


class RoWordEmbeddings(tf.keras.initializers.Initializer):
    def __init__(self, lexicon: Lex) -> None:
        self._lexicon = lexicon

    def __call__(self, shape, dtype=None, **kwargs):
        """Called only when training."""

        if len(shape) == 2:
            if shape[0] == self._wembvsz:
                return tf.constant(np.asarray(self._wembmat), dtype=dtype)
            else:
                return tf.constant(np.asarray(self._wembmat).transpose(), dtype=dtype)
            # end if
        else:
            return tf.keras.initializers.GlorotUniform()(shape=shape)
        # end if

    def load_ids(self):
        """Call this at runtime, together with Keras model loading."""
        self._wembvoc = {}

        if os.path.exists(EMBEDDING_VOCABULARY_FILE):
            with open(EMBEDDING_VOCABULARY_FILE, mode='r', encoding='utf-8') as f:
                first_line = True

                for line in f:
                    line = line.rstrip()

                    if first_line:
                        self._wembdim = int(line)
                        first_line = False
                        continue
                    # end if

                    parts = line.split()
                    word = parts[0]
                    wid = int(parts[1])
                    self._wembvoc[word] = wid
                # end for
            # end with
        # end if

        self._wembvsz = len(self._wembvoc)

    def load_word_embeddings(self, word_list: set):
        self._wembdim = self._lexicon.get_word_embeddings_size()
        self._wembvoc = {}
        self._wembmat = []
        self._wembmat.append([0] * self._wembdim)
        self._wembvoc[_zero_word] = 0
        # The vector for the unknown word
        self._wembmat.append([0.1] * self._wembdim)
        self._wembvoc[_unk_word] = 1
        word_id = 2

        for word in sorted(word_list):
            wwe = self._lexicon.get_word_embeddings_exact(word)

            if wwe:
                self._wembvoc[word] = word_id
                self._wembmat.append(wwe)
            else:
                lc_word = word.lower()
                wwe = self._lexicon.get_word_embeddings_exact(lc_word)

                if wwe:
                    self._wembvoc[lc_word] = word_id
                    self._wembmat.append(wwe)
                else:
                    self._wembvoc[word] = word_id
                    self._wembmat.append([0.5] * self._wembdim)
                # end if
            # end if

            word_id += 1
        # end for

        self._wembvsz = word_id

        # Add extra words to this vocabulary and save them
        with open(EMBEDDING_VOCABULARY_FILE, mode='w', encoding='utf-8') as f:
            print("{0!s}".format(self._wembvsz), file=f)

            for word in sorted(self._wembvoc.keys()):
                print("{0}\t{1!s}".format(word, self._wembvoc[word]), file=f)
            # end for
        # end with

    def get_word_id(self, word: str) -> int:
        if word in self._wembvoc:
            return self._wembvoc[word]
        elif word.lower() in self._wembvoc:
            return self._wembvoc[word.lower()]
        else:
            return self._wembvoc[_unk_word]
        # end if

    def get_vector_length(self) -> int:
        return self._wembdim

    def get_vocabulary_size(self) -> int:
        return self._wembvsz


class CRFModel(tf.keras.Model):

    def __init__(self, tagset_size, prev_inputs: list, prev_outputs: list, layer_input):
        """Takes all of the inputs from the LM model and its outputs, and builds a CRF model
        with `inputs=lm_inputs` and `outputs=lm_outputs + CRF outputs`"""

        masked_input = prev_inputs[-1]

        # Build functional model
        crf = tfa.layers.CRF(
            units=tagset_size,
            chain_initializer=tf.keras.initializers.Orthogonal(),
            use_boundary=True,
            boundary_initializer=tf.keras.initializers.Zeros(),
            use_kernel=True,
            name='crf_layer')

        (decode_sequence, potentials, sequence_length, kernel) = crf(
            inputs=layer_input, mask=masked_input)

        # Set name for outputs
        decode_sequence = tf.keras.layers.Lambda(
            lambda x: x, name='decode_sequence')(decode_sequence)
        potentials = tf.keras.layers.Lambda(
            lambda x: x, name='potentials')(potentials)
        sequence_length = tf.keras.layers.Lambda(
            lambda x: x, name='sequence_length')(sequence_length)
        kernel = tf.keras.layers.Lambda(lambda x: x, name='kernel')(kernel)

        super().__init__(
            inputs=prev_inputs,
            outputs=prev_outputs + [decode_sequence, potentials, sequence_length, kernel])

        self._custom_metrics_train = {
            'msd_enc': tf.keras.metrics.BinaryAccuracy(name='encoding_acc'),
            'msd_cls': tf.keras.metrics.CategoricalAccuracy(name='rnn_acc'),
            'decode_sequence': tf.keras.metrics.Accuracy(name='viterbi_acc')
        }
        self._loss_tracker_train = {
            'crf_loss': tf.keras.metrics.Mean(name="crf_loss"),
            'dec_loss': tf.keras.metrics.Mean(name="dec_loss")
        }

        self._custom_metrics_test = {
            'msd_enc': tf.keras.metrics.BinaryAccuracy(name='encoding_acc'),
            'msd_cls': tf.keras.metrics.CategoricalAccuracy(name='rnn_acc'),
            'decode_sequence': tf.keras.metrics.Accuracy(name='viterbi_acc')
        }
        self._loss_tracker_test = {
            'crf_loss': tf.keras.metrics.Mean(name="crf_loss"),
            'dec_loss': tf.keras.metrics.Mean(name="dec_loss")
        }

    def train_step(self, data):
        x_lex = data[0][0]
        x_emb = data[0][1]
        x_ctx = data[0][2]
        z = data[0][3]
        # y has shape (batch_size, max_seq_len) of type tf.int32
        # with indices of ground truth label on [i,j]
        y_enc = data[1]['msd_enc']
        y_cls = data[1]['msd_cls']
        y_crf = data[1]['decode_sequence']

        bin_cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='bin_CE')
        cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='cat_CE')
        bin_metric = self._custom_metrics_train['msd_enc']
        oneh_metric = self._custom_metrics_train['msd_cls']
        vit_metric = self._custom_metrics_train['decode_sequence']

        with tf.GradientTape(persistent=True) as tape:
            msd_enc, msd_cls, decode_sequence, potentials, sequence_length, kernel = \
                self(inputs=[x_lex, x_emb, x_ctx, z], mask=z, training=True)
            crf_loss = - \
                tfa.text.crf_log_likelihood(
                    potentials, y_crf, sequence_length, kernel)[0]
            crf_loss = tf.reduce_mean(crf_loss)
            bce_loss = bin_cross_entropy(y_enc, msd_enc)
            cce_loss = cat_cross_entropy(y_cls, msd_cls)
            loss_dec = bce_loss + cce_loss + sum(self.losses)
            loss_crf = bce_loss + crf_loss + sum(self.losses)
        # end with

        dec_variables = []
        crf_variables = []

        # Establish groups of variables for differentiation
        for v in self.trainable_variables:
            if v.name.startswith('embedding') or \
                v.name.startswith('bidirectional') or \
                v.name.startswith('msd_enc'):
                dec_variables.append(v)
                crf_variables.append(v)
            elif v.name.startswith('bidirectional_1') or \
                v.name.startswith('dense_linear'):
                dec_variables.append(v)
            elif v.name.startswith('chain_kernel') or \
                v.name.startswith('left_boundary') or \
                v.name.startswith('right_boundary') or \
                v.name.startswith('crf_layer'):
                crf_variables.append(v)
            # end if
        # end for

        self.optimizer.minimize(loss_dec, dec_variables, tape=tape)
        self.optimizer.minimize(loss_crf, crf_variables, tape=tape)
        del tape

        # Update metrics (includes the metric that tracks the loss)
        self._loss_tracker_train['dec_loss'].update_state(loss_dec)
        self._loss_tracker_train['crf_loss'].update_state(loss_crf)
        bin_metric.update_state(y_enc, msd_enc)
        oneh_metric.update_state(y_cls, msd_cls)
        vit_metric.update_state(y_crf, decode_sequence)

        return {
            self._loss_tracker_train['dec_loss'].name: self._loss_tracker_train['dec_loss'].result(),
            self._loss_tracker_train['crf_loss'].name: self._loss_tracker_train['crf_loss'].result(),
            bin_metric.name: bin_metric.result(),
            oneh_metric.name: oneh_metric.result(),
            vit_metric.name: vit_metric.result()
        }

    def test_step(self, data):
        x_lex = data[0][0]
        x_emb = data[0][1]
        x_ctx = data[0][2]
        z = data[0][3]
        # y has shape (batch_size, max_seq_len) of type tf.int32
        # with indices of ground truth label on [i,j]
        y_enc = data[1]['msd_enc']
        y_cls = data[1]['msd_cls']
        y_crf = data[1]['decode_sequence']

        bin_cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='bin_CE')
        cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='cat_CE')
        bin_metric = self._custom_metrics_test['msd_enc']
        oneh_metric = self._custom_metrics_test['msd_cls']
        vit_metric = self._custom_metrics_test['decode_sequence']

        msd_enc, msd_cls, decode_sequence, potentials, sequence_length, kernel = \
            self(inputs=[x_lex, x_emb, x_ctx, z], mask=z, training=False)
        crf_loss = - \
            tfa.text.crf_log_likelihood(
                potentials, y_crf, sequence_length, kernel)[0]
        crf_loss = tf.reduce_mean(crf_loss)
        bce_loss=bin_cross_entropy(y_enc, msd_enc)
        cce_loss=cat_cross_entropy(y_cls, msd_cls)
        loss_dec = bce_loss + cce_loss + sum(self.losses)
        loss_crf = bce_loss + crf_loss + sum(self.losses)


        # Update metrics (includes the metric that tracks the loss)
        self._loss_tracker_test['dec_loss'].update_state(loss_dec)
        self._loss_tracker_test['crf_loss'].update_state(loss_crf)
        bin_metric.update_state(y_enc, msd_enc)
        oneh_metric.update_state(y_cls, msd_cls)
        vit_metric.update_state(y_crf, decode_sequence)

        return {
            self._loss_tracker_test['dec_loss'].name: self._loss_tracker_test['dec_loss'].result(),
            self._loss_tracker_test['crf_loss'].name: self._loss_tracker_test['crf_loss'].result(),
            bin_metric.name: bin_metric.result(),
            oneh_metric.name: oneh_metric.result(),
            vit_metric.name: vit_metric.result()
        }


class RoPOSTagger(object):
    """This class will do MSD POS tagging for Romanian.
    It will train/test the DNN models and also, given a string of Romanian text,
    it will split it in sentences, POS tag each sentence and return the list."""

    # How many words in a window to consider when constructing a sample.
    # This is the Tx value in the Deep Learning course.
    # Set to 0 to estimate it as the average sentence length in the
    # training set.
    _conf_maxseqlen = 30
    # How much (%) to retain from the train data as dev/test sets
    _conf_dev_percent = 0.1
    # No test, for now, look at values on dev
    _conf_test_percent = 0.0
    # RNN state size
    _conf_rnn_size_1 = 128
    _conf_rnn_size_2 = 128
    _conf_epochs = 10
    _conf_with_tiered_tagging = True
    # Can be one of the following (see RoPOSTagger._run_sentence()):
    # - 'add': add the probabilities for each MSD that was assigned at position i, in each rolling window
    # - 'max': only keep the MSD with the highest probability at position i, from each rolling window
    # - 'cnt': add 1 to each different MSD that was assigned at position i, in each rolling window (majority voting)
    _conf_run_strategy = 'add'

    def __init__(self, splitter: RoSentenceSplitter):
        """Takes a trained instance of the RoSentenceSplitter object."""
        self._splitter = splitter
        self._tokenizer = splitter.get_tokenizer()
        self._uniprops = CharUni()
        self._lexicon = splitter.get_lexicon()
        self._msd = self._lexicon.get_msd_object()
        self._rofeatures = RoFeatures(self._lexicon)
        self._romorphology = RoInflect(self._lexicon)
        self._romorphology.load()
        self._wordembeddings = RoWordEmbeddings(self._lexicon)
        self._datavocabulary = set()
        self._maxseqlen = RoPOSTagger._conf_maxseqlen

    @staticmethod
    def _select_tf_device() -> str:
        """I always get OOM errors when training the CRF layer.
        So, choose some other device for training it."""

        physical_devices = tf.config.list_physical_devices('GPU')
        device = "/device:CPU:0"

        if RoPOSTagger._conf_maxseqlen <= 30 and \
            RoPOSTagger._conf_rnn_size_1 <= 128 and \
            RoPOSTagger._conf_rnn_size_2 <= 128:
            device = "/device:GPU:0"
        # end if

        if len(physical_devices) > 1:
            device = "/device:GPU:1"
        # end if

        print(stack()[0][3] + ": RoPOSTagger training will run on '{0}'".format(device),
                file=sys.stderr, flush=True)

        return device

    def _save(self):
        self._uniprops.save_unicode_props(TAGGER_UNICODE_PROPERTY_FILE)
        self._model.save(TAGGER_MODEL_FOLDER, overwrite=True)

    def load(self):
        self._uniprops.load_unicode_props(TAGGER_UNICODE_PROPERTY_FILE)
        self._wordembeddings.load_ids()
        self._model = tf.keras.models.load_model(TAGGER_MODEL_FOLDER)

    def train(self,
        data_sentences: list = [],
        train_sentences: list = [],
        dev_sentences: list = [],
        test_sentences: list = []):
        # Normalize the whole data vocabulary and load external word embeddings
        self._normalize_vocabulary()
        self._wordembeddings.load_word_embeddings(self._datavocabulary)

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

            (train_sentences, train_examples, dev_sentences, dev_examples, test_sentences,
             test_examples) = self._build_train_samples(data_sentences)
        elif not data_sentences and \
                train_sentences and dev_sentences and test_sentences:
            (train_examples, dev_examples, test_examples) = self._build_train_samples_already_split(
                train_sentences, dev_sentences, test_sentences)
        # end if

        print(stack()[0][3] + ": got train set with {0!s} samples".format(
            len(train_examples)), file=sys.stderr, flush=True)
        print(stack()[0][3] + ": got train set with {0!s} sentences".format(
            len(train_sentences)), file=sys.stderr, flush=True)
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

        # 4. Get train/dev/test numpy tensors
        def _generate_tensors(ml_type: str, examples: list) -> tuple:
            print(stack()[0][3] + ": building ENC/CLS {0} tensors".format(ml_type),
                  file=sys.stderr, flush=True)

            (x_lex, x_emb, x_ctx, y_enc, y_cls, y_crf, z_msk) = self._build_model_io_tensors(examples)

            print(stack()[0][3] + ": {0} x_lex.shape is {1!s}".format(
                ml_type, x_lex.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} x_emb.shape is {1!s}".format(
                ml_type, x_emb.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} x_ctx.shape is {1!s}".format(
                ml_type, x_ctx.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} ENC y.shape is {1!s}".format(
                ml_type, y_enc.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} CLS y.shape is {1!s}".format(
                ml_type, y_cls.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} CRF y.shape is {1!s}".format(
                ml_type, y_crf.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} MASK z.shape is {1!s}".format(
                ml_type, z_msk.shape), file=sys.stderr, flush=True)

            return (x_lex, x_emb, x_ctx, y_enc, y_cls, y_crf, z_msk)
        # end def

        (x_lex_train, x_emb_train, x_ctx_train, y_train_enc,
         y_train_cls, y_train_crf, z_train_mask) = _generate_tensors('train', train_examples)
        (x_lex_dev, x_emb_dev, x_ctx_dev, y_dev_enc,
         y_dev_cls, y_dev_crf, z_dev_mask) = _generate_tensors('dev', dev_examples)

        # 4.1 Save RoInflect cache file for faster startup next time
        self._romorphology.save_cache()

        lex_input_dim = x_lex_train.shape[2]
        ctx_input_dim = x_ctx_train.shape[2]
        encoding_dim = y_train_enc.shape[2]
        output_msd_dim = self._msd.get_output_vector_size()
        output_ctg_dim = self._msd.get_ctag_output_vector_size()

        with tf.device(RoPOSTagger._select_tf_device()):
            # 5. Creating the language model
            self._model = self._build_keras_model(
                lex_input_dim,
                ctx_input_dim,
                self._wordembeddings.get_vocabulary_size(),
                self._wordembeddings.get_vector_length(),
                encoding_dim, output_msd_dim, output_ctg_dim,
                drop_prob=0.25
            )

            # 6. Print model summary
            self._model.summary()

            # 7. Train the LM model
            self._train_keras_model(
                train=(x_lex_train, x_emb_train, x_ctx_train, y_train_enc,
                    y_train_cls, y_train_crf, z_train_mask),
                dev=(x_lex_dev, x_emb_dev, x_ctx_dev, y_dev_enc,
                    y_dev_cls, y_dev_crf, z_dev_mask),
                gold_sentences=dev_sentences
            )

            # 8. Saving the model
            self._save()
        # end with device

    def _get_predicted_msd(self, y_pred: np.ndarray) -> tuple:
        best_idx = np.argmax(y_pred)
        msd = self._msd.idx_to_msd(best_idx)
        msd_p = y_pred[best_idx]

        return (msd, msd_p)

    def _fixed_tag_assignments(self, word: str) -> str:
        """This method will assigned a fixed MSD depending on the shape of word.
        For instance, do not let the neural network get punctuation or numbers
        tagged wrongly."""

        if word in MSD.punct_msd_inventory:
            return MSD.punct_msd_inventory[word]
        elif MSD.punct_patt.match(word):
            return MSD.unknown_punct_msd
        # end punctuation

        if Lex.number_pattern.match(word):
            return MSD.number_msd
        elif Lex.bullet_number_pattern.match(word):
            return MSD.bullet_number_msd
        # end numbers

        # TODO: add a foreign name -> Np mapping here.
        # Learn to detect foreign names.

        return ''

    def _most_prob_msd(self, word: str, y_pred: np.ndarray) -> tuple:
        # 0. Get fixed tag assignment, if any
        fixed_msd = self._fixed_tag_assignments(word)

        if fixed_msd:
            return (fixed_msd, 1.)
        # end if

        # 1. Get the model predicted MSD
        (best_pred_msd, best_pred_msd_p) = self._get_predicted_msd(y_pred)

        if best_pred_msd.startswith(MSD.unknown_label) or \
            best_pred_msd.startswith(MSD.unknown_punct_msd):
            # Do not touch punctuation and X
            return (best_pred_msd, best_pred_msd_p)
        # end if

        # 2. Get the extended word ambiguity class, as the predicted MSD may
        # be outside this set.
        word_msds = set()

        if self._lexicon.is_lex_word(word):
            word_msds.update(self._lexicon.get_word_ambiguity_class(word))
        # end if

        aclass = self._romorphology.ambiguity_class(word)

        if not aclass:
            aclass = self._lexicon.get_unknown_ambiguity_class(word)
        # end if

        word_msds.update(aclass)

        if best_pred_msd in word_msds:
            # 3. Model predicted MSD is in the extended ambiguity class. Hurray!
            return (best_pred_msd, best_pred_msd_p)
        # end if
        
        # 4. Model predicted MSD MSD is not in the extended ambiguity class.
        # Choose the highest probability MSD from the extended ambiguity class.
        best_acls_msd_p = 0.
        best_acls_msd = ''

        for msd in word_msds:
            idx = self._msd.msd_to_idx(msd)
            msd_p = y_pred[idx]

            if msd_p > best_acls_msd_p:
                best_acls_msd_p = msd_p
                best_acls_msd = msd
            # end if
        # end for extended ambiguity class

        print(stack()[0][3] + \
            ": word '{0}' -> got suboptimal MSD [{1}/{2:.2f}] vs. optimal [{3}/{4:.2f}]".
            format(word, best_acls_msd, best_acls_msd_p, best_pred_msd, best_pred_msd_p),
            file=sys.stderr, flush=True)

        return (best_acls_msd, best_acls_msd_p)

    def _tiered_tagging(self, word: str, ctag: str) -> list:
        """This implements Tufiș and Dragomirescu's (2004) tiered tagging concept that
        deterministically retrieves the more complex POS label, given the `word`, compressed label `ctag` and a lexicon.
        See here: https://aclanthology.org/L04-1158/
        Returns the corresponding MSD of `word`.
        """
        
        fixed_msd = self._fixed_tag_assignments(word)

        if fixed_msd:
            return [fixed_msd]
        # end if

        # Mapped class of MSDs
        mclass = self._msd.ctag_to_possible_msds(ctag)
        lex_aclass = []

        if self._lexicon.is_lex_word(word):
            # Case 1.1: word is in lexicon (handles exact match and lower case match).
            lex_aclass = self._lexicon.get_word_ambiguity_class(word)
        # end if

        infl_aclass = []

        if MSD.content_word_ctag_pattern.match(ctag):
            # Case 1.2: word is not in lexicon. Try and determine its ambiguity class.
            infl_aclass = self._romorphology.ambiguity_class(word)

            if not infl_aclass:
                # Default to the lexicon-based one.
                infl_aclass = self._lexicon.get_unknown_ambiguity_class(word)
            # end if
        # end if

        both_aclass = set(lex_aclass).union(set(infl_aclass))
        
        # Reduced class of MSDs
        return list(both_aclass.intersection(set(mclass)))

    def _run_sentence(self, sentence, tiered_tagging: bool = True, strategy: str = 'max', debug_fh = None):
        """Strategy can be `'add'` or `'max'`. When `'add'`ing contributions from the LM and CRF layers,
        a dict of predicted MSDs is kept at each position in the sentence and we add the probabilities
        coming from each rolling window, at position `i`. The best MSD wins at position `i`.

        We can also keep the winning, predicted MSD (by its `'max'` probability) at each position `i` in the sentence,
        with each rolling window. """
        # 1. Build the fixed-length samples from the input sentence
        sent_samples = self._build_samples([sentence])

        # 2. Get the input tensors
        (x_lex_run, x_emb_run, x_ctx_run, _, _, _, z_mask) = self._build_model_io_tensors(
            sent_samples, runtime=True)
        # end if

        # 3. Use the model to get predicted MSDs
        # y_pred_cls is the POS tagger sequence with RNNs, one-hot
        # viterbi_sequence is the POS tagger sequence with CRFs, CTAG index
        (y_pred_enc, y_pred_cls, viterbi_sequence, pot, seq_len, krn) = self._model.predict(
            x=[x_lex_run, x_emb_run, x_ctx_run, z_mask])

        assert y_pred_cls.shape[0] == viterbi_sequence.shape[0]
        assert y_pred_cls.shape[0] == len(sent_samples)
        assert y_pred_cls.shape[1] == viterbi_sequence.shape[1]
        assert y_pred_cls.shape[1] == len(sent_samples[0])

        tagged_sentence = []
        debug_info = []

        def _add_debug_info(info: dict, msd: str, msd_p: float):
            if msd not in info:
                info[msd] = {
                    'cnt': 1, 'pm': msd_p, 'max': msd_p}
            else:
                info[msd]['cnt'] += 1
                info[msd]['pm'] += msd_p

                if msd_p > info[msd]['max']:
                    info[msd]['max'] = msd_p
                # end if
            # end if
        # end def

        def _find_debug_highlights(info):
            max_cnt = 0
            max_cnt_msd = []
            max_pmass = 0.
            max_pmass_msd = []
            max_maxp = 0.
            max_maxp_msd = []

            for msd in info:
                minfo = info[msd]

                if minfo['cnt'] > max_cnt:
                    max_cnt = minfo['cnt']
                    max_cnt_msd = [msd]
                elif minfo['cnt'] == max_cnt:
                    max_cnt_msd.append(msd)
                # end if

                if minfo['pm'] > max_pmass:
                    max_pmass = minfo['pm']
                    max_pmass_msd = [msd]
                elif isclose(minfo['pm'], max_pmass):
                    max_pmass_msd.append(msd)
                # end if

                if minfo['max'] > max_maxp:
                    max_maxp = minfo['max']
                    max_maxp_msd = [msd]
                elif isclose(minfo['max'], max_maxp):
                    max_maxp_msd.append(msd)
                # end if
            # end all MSDs

            return (max_cnt_msd, max_pmass_msd, max_maxp_msd)
        # end def

        for i in range(len(sent_samples)):
            sample = sent_samples[i]
            sentence_done = False

            for j in range(len(sample)):
                if (i + j < len(sentence)):
                    word = sentence[i + j][0]

                    if len(tagged_sentence) <= i + j:
                        if strategy == 'add' or strategy == 'cnt':
                            tagged_sentence.append((word, {}))
                        elif strategy == 'max':
                            tagged_sentence.append((word, []))
                        # end if
                        debug_info.append({'CRF': {}, 'RNN': {}})
                    # end if

                    j_debug_info = debug_info[i + j]
                    y1 = y_pred_cls[i, j, :]
                    (msd_rnn, msd_rnn_p) = self._most_prob_msd(word, y1)
                    current_msd_options = [(msd_rnn, msd_rnn_p)]

                    _add_debug_info(j_debug_info['RNN'], msd_rnn, msd_rnn_p)
                    
                    if tiered_tagging:
                        # Index of the most probable CTAG
                        y2 = viterbi_sequence[i, j]
                        ctag = self._msd.idx_to_ctag(y2)
                        ctag_msds = self._tiered_tagging(word, ctag)

                        if not ctag_msds:
                            _add_debug_info(j_debug_info['CRF'], ctag, 1.)
                        else:
                            for m in ctag_msds:
                                mp = 1. / len(ctag_msds)
                                _add_debug_info(j_debug_info['CRF'], m, mp)
                                current_msd_options.append((m, mp))
                            # end for
                        # end if
                    # end if with tiered tagging

                    ij_msd_best = tagged_sentence[i + j][1]

                    if strategy == 'add':
                        for msd, msd_p in current_msd_options:
                            if msd in ij_msd_best:
                                ij_msd_best[msd] += msd_p
                            else:
                                ij_msd_best[msd] = msd_p
                            # end if
                        # end for
                    elif strategy == 'cnt':
                        for msd, msd_p in current_msd_options:
                            if msd in ij_msd_best:
                                ij_msd_best[msd] += 1
                            else:
                                ij_msd_best[msd] = 1
                            # end if
                        # end for
                    elif strategy == 'max':
                        for msd, msd_p in current_msd_options:
                            if ij_msd_best:
                                if msd_p > ij_msd_best[1]:
                                    ij_msd_best[0] = msd
                                    ij_msd_best[1] = msd_p
                                # end if
                            else:
                                ij_msd_best.append(msd)
                                ij_msd_best.append(msd_p)
                            # end if
                        # end for
                    # end if strategy
                else:
                    sentence_done = True
                    break
                # end if
            # end for j

            if sentence_done:
                break
            # end if
        # end for i

        assert len(tagged_sentence) == len(sentence)

        if debug_fh:
            counts  = {
                'rnn_cnt_correct': 0,
                'rnn_pmass_correct': 0,
                'rnn_maxp_correct': 0,
                'crf_cnt_correct': 0,
                'crf_pmass_correct': 0,
                'crf_maxp_correct': 0
            }

            print(">>>>>",
                  file=debug_fh, flush=True)

            for i in range(len(debug_info)):
                word = sentence[i][0]
                gold_msd = sentence[i][1]
                rnn_info = debug_info[i]['RNN']
                (rnn_msd_cnt, rnn_msd_pmass, rnn_msd_maxp) = _find_debug_highlights(rnn_info)
                crf_info = debug_info[i]['CRF']
                (crf_msd_cnt, crf_msd_pmass, crf_msd_maxp) = _find_debug_highlights(crf_info)
                rnn_correct = 0
                crf_correct = 0

                if gold_msd in rnn_msd_cnt:
                    counts['rnn_cnt_correct'] += 1
                    rnn_correct += 1
                # end if

                if gold_msd in rnn_msd_pmass:
                    counts['rnn_pmass_correct'] += 1
                    rnn_correct += 1
                # end if

                if gold_msd in rnn_msd_maxp:
                    counts['rnn_maxp_correct'] += 1
                    rnn_correct += 1
                # end if

                if gold_msd in crf_msd_cnt:
                    counts['crf_cnt_correct'] += 1
                    crf_correct += 1
                # end if

                if gold_msd in crf_msd_pmass:
                    counts['crf_pmass_correct'] += 1
                    crf_correct += 1
                # end if

                if gold_msd in crf_msd_maxp:
                    counts['crf_maxp_correct'] += 1
                    crf_correct += 1
                # end if

                print("{0}\t{1}".format(word, gold_msd), file=debug_fh, flush=True, end='\t')

                if rnn_correct == 3:
                    print("RNN=", file=debug_fh, flush=True, end='\t')
                elif rnn_correct > 0:
                    print("RNN~ c={0};pm={1};mx={2}".format(
                        ','.join(rnn_msd_cnt), ','.join(rnn_msd_pmass), ','.join(rnn_msd_maxp)), file=debug_fh, flush=True, end='\t')
                else:
                    print("RNN! c={0};pm={1};mx={2}".format(
                        ','.join(rnn_msd_cnt), ','.join(rnn_msd_pmass), ','.join(rnn_msd_maxp)), file=debug_fh, flush=True, end='\t')
                # end if

                if crf_correct == 3:
                    print("CRF=", file=debug_fh, flush=True)
                elif crf_correct > 0:
                    print("CRF~ c={0};pm={1};mx={2}".format(
                        ','.join(crf_msd_cnt), ','.join(crf_msd_pmass), ','.join(crf_msd_maxp)), file=debug_fh, flush=True)
                else:
                    print("CRF! c={0};pm={1};mx={2}".format(
                        ','.join(crf_msd_cnt), ','.join(crf_msd_pmass), ','.join(crf_msd_maxp)), file=debug_fh, flush=True)
                # end if
            # end for i

            best_performance = 0
            best_methods = []

            for pm in counts:
                if counts[pm] > best_performance:
                    best_performance = counts[pm]
                    best_methods = [pm]
                elif counts[pm] == best_performance:
                    best_methods.append(pm)
                # end if
            # end for

            acc = float(best_performance) / len(debug_info)

            print("Best performances (Acc = {0:.5f}): {1}".format(acc,
                ', '.join(best_methods)), file=debug_fh, flush=True)
            print("<<<<<",
                  file=debug_fh, flush=True)
        # end if debug

        tagged_sentence2 = []

        for i in range(len(tagged_sentence)):
            word, choice = tagged_sentence[i]
            best_msd = '?'
            # Not a probability but an additive score
            # The larger, the better
            best_msd_score = 0.

            if strategy == 'add' or strategy == 'cnt':
                for m in choice:
                    if choice[m] > best_msd_score:
                        best_msd_score = choice[m]
                        best_msd = m
                    # end if
                # end for
            elif strategy == 'max':
                best_msd = choice[0]
                best_msd_score = choice[1]
            # end if

            tagged_sentence2.append((word, best_msd, best_msd_score))
        # end for

        return tagged_sentence2

    def _train_keras_model(self, train: tuple, dev: tuple, gold_sentences: list):
        # Compile model
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self._model.compile(
            optimizer=opt,
            # Set this to true for Python debugging
            run_eagerly=None
        )

        x_lex_train = train[0]
        x_emb_train = train[1]
        x_ctx_train = train[2]
        x_mask_train = train[6]
        y_train = {
            'msd_enc': train[3],
            'msd_cls': train[4],
            'decode_sequence': train[5]
        }

        x_lex_dev = dev[0]
        x_emb_dev = dev[1]
        x_ctx_dev = dev[2]
        x_mask_dev = dev[6]
        y_dev = {
            'msd_enc': dev[3],
            'msd_cls': dev[4],
            'decode_sequence': dev[5]
        }

        # Fit model
        acc_callback = AccCallback(
            self, gold_sentences, RoPOSTagger._conf_epochs,
            RoPOSTagger._conf_with_tiered_tagging, RoPOSTagger._conf_run_strategy)
        self._model.fit(
            x=[x_lex_train, x_emb_train, x_ctx_train, x_mask_train],
            y=y_train,
            epochs=RoPOSTagger._conf_epochs, batch_size=16, shuffle=True,
            validation_data=([x_lex_dev, x_emb_dev, x_ctx_dev, x_mask_dev], y_dev),
            callbacks=[acc_callback]
        )
    
    def _build_keras_model(self,
                           lex_input_vector_size: int,
                           ctx_input_vector_size: int,
                           word_embeddings_voc_size: int,
                           word_embeddings_proj_size: int,
                           msd_encoding_vector_size: int,
                           output_msd_size: int,
                           output_ctg_size: int,
                           drop_prob: float = 0.33
                           ) -> tf.keras.Model:
        # Inputs
        x_lex = tf.keras.layers.Input(shape=(self._maxseqlen,
                                             lex_input_vector_size), dtype='float32', name="word_lexical_input")
        x_ctx = tf.keras.layers.Input(shape=(self._maxseqlen,
                                             ctx_input_vector_size), dtype='float32', name="word_context_input")
        x_emb = tf.keras.layers.Input(
            shape=(self._maxseqlen,), dtype='int32', name="word_embedding_input")
        z_mask = tf.keras.layers.Input(shape=(self._maxseqlen,), dtype='bool', name="masked_input")
        # End inputs

        # MSD encoding
        l_emb = tf.keras.layers.Embedding(
            word_embeddings_voc_size, word_embeddings_proj_size,
            embeddings_initializer=self._wordembeddings, mask_zero=True,
            input_length=self._maxseqlen)(x_emb)
        l_lex_emb_conc = tf.keras.layers.Concatenate()([x_lex, l_emb])
        l_bd_gru_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(RoPOSTagger._conf_rnn_size_1, return_sequences=True))(l_lex_emb_conc)
        l_drop_1 = tf.keras.layers.Dropout(drop_prob)(l_bd_gru_1)
        o_msd_enc = tf.keras.layers.Dense(
            msd_encoding_vector_size, activation='sigmoid', name='msd_enc')(l_drop_1)
        # End MSD encoding

        l_drop_2 = tf.keras.layers.Dropout(drop_prob)(o_msd_enc)
        l_msd_enc_ctx_conc = tf.keras.layers.Concatenate(
            name='enc_states')([l_drop_2, x_ctx])

        # Language model
        l_bd_gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            RoPOSTagger._conf_rnn_size_2, return_sequences=True))(l_drop_2)
        l_drop_3 = tf.keras.layers.Dropout(drop_prob)(l_bd_gru_2)
        l_bd_gru2_ctx_conc = tf.keras.layers.Concatenate(
            name='lm_states')([l_drop_3, x_ctx])
        l_dense_1 = tf.keras.layers.Dense(
            output_msd_size, name="dense_linear")(l_bd_gru2_ctx_conc)
        o_msd_cls = tf.keras.layers.Activation(
            'softmax', name='msd_cls')(l_dense_1)
        # End language model

        return CRFModel(
            output_ctg_size,
            [x_lex, x_emb, x_ctx, z_mask],
            [o_msd_enc, o_msd_cls],
            l_msd_enc_ctx_conc
        )

    def _build_model_io_tensors(self, data_samples, runtime: bool = False) -> tuple:
        # No of examples
        m = len(data_samples)
        # This should be equal to self._maxseqlen
        tx = len(data_samples[0])
        # assert Tx == self._maxseqlen
        # This is the size of the input vector
        n = -1

        ### Inputs
        # Lexical tensor
        xlex_tensor = None
        # Word embeddings tensor
        xemb_tensor = np.empty((m, tx), dtype=np.int32)
        # Externally computed contextual features
        xctx_tensor = np.empty(
            (m, tx, len(RoFeatures.romanian_pos_tagging_features)), dtype=np.float32)

        ### Ground truth outputs
        y_tensor_crf = np.empty((m, tx), dtype=np.int32)
        z_tensor_mask = np.empty((m, tx), dtype=bool)
        # Ys for the Keras MSD encoding part
        y_tensor_enc = np.empty(
            (m, tx, self._msd.get_input_vector_size()), dtype=np.float32)
        # Ys for the Keras MSD classification part
        y_tensor_cls = np.empty(
            (m, tx, self._msd.get_output_vector_size()), dtype=np.float32)

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
                ctag = self._msd.msd_to_ctag(msd)

                if not self._msd.is_valid_ctag(ctag):
                    print(stack()[0][3] + ": unknown CTAG [{0}] for MSD [{1}]".format(
                        ctag, msd), file=sys.stderr, flush=True)
                # end if

                feats = parts[2]
                tlabel = self._tokenizer.tag_word(word)
                z_mask = True

                if not runtime:
                    y_in = self._msd.msd_input_vector(msd)
                    y_out = self._msd.msd_reference_vector(msd)
                    y_out_idx = self._msd.ctag_to_idx(ctag)
                else:
                    y_in = self._msd.get_x_input_vector()
                    y_out = self._msd.get_x_reference_vector()
                    y_out_idx = self._msd.ctag_to_idx('X')
                # end if

                label_features = self._tokenizer.get_label_features(tlabel)
                uni_features = self._uniprops.get_unicode_features(word)
                lexical_features = self._get_lexical_features_for_pos_tagging(
                    word)

                # This is the featurized version of a word in the sequence
                x_lex = np.concatenate(
                    (label_features, uni_features, lexical_features))

                if word == _zero_word:
                    x_lex = np.zeros(x_lex.shape, dtype=np.float32)
                    z_mask = False
                # end if

                if n == -1:
                    n = x_lex.shape[0]
                    xlex_tensor = np.empty((m, tx, n), dtype=np.float32)
                # end if

                # Computing id for word
                x_wid = self._wordembeddings.get_word_id(word)
                # Computing external features for word
                x_ctx = self._rofeatures.get_context_feature_vector(feats)

                xlex_tensor[i, j, :] = x_lex
                xemb_tensor[i, j] = x_wid
                xctx_tensor[i, j, :] = x_ctx
                y_tensor_crf[i, j] = y_out_idx
                z_tensor_mask[i, j] = z_mask
                y_tensor_enc[i, j, :] = y_in
                y_tensor_cls[i, j, :] = y_out
            # end j
        # end i

        return (xlex_tensor, xemb_tensor, xctx_tensor, y_tensor_enc, y_tensor_cls, y_tensor_crf, z_tensor_mask)

    def _build_samples(self, sentences: list) -> list:
        all_samples = []

        # Assemble examples with required, fixed length
        for sentence in sentences:
            # Make all sentences have RoPOSTag.maxSeqLength at least.
            p_sentence = self._pad_sentence(sentence)

            for i in range(len(p_sentence)):
                left = i
                right = i + self._maxseqlen

                if right > len(p_sentence):
                    break
                # end if

                all_samples.append(p_sentence[left:right])
            # end for i
        # end all sentences

        return all_samples

    def _pad_sentence(self, sentence: list) -> list:
        """If the sentence has less tokens than the trained
        seqence length, pad it with the last word, at the right,
        to reach RoPOSTag.maxSeqLength."""
        padded_sentence = sentence[:]

        while len(padded_sentence) < self._maxseqlen:
            padded_sentence.append((_zero_word, 'X', []))
        # end with

        return padded_sentence

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

        sentence_lengths = []

        # Find the maximum sentence length in train
        for sentence in train_sentences:
            sentence_lengths.append(len(sentence))
        # end for

        slarr = np.asarray(sentence_lengths, dtype=np.float32)
        sl_max = np.max(slarr, keepdims=False)

        print(stack()[0][3] + ": maximum sentence length in training set is {0:.2f}".format(sl_max), file=sys.stderr, flush=True)

        sl_mean = np.mean(slarr)
        sl_stddev = np.std(slarr)

        if self._maxseqlen == 0:
            self._maxseqlen = int(sl_mean)

            print(stack()[0][3] + ": determined {0!s} as the maximum sentence length (mean = {1:.5f}, stddev = {2:.5f})".format(
                self._maxseqlen, sl_mean, sl_stddev), file=sys.stderr, flush=True)
        # end if

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

        sentence_lengths = []

        # Find the maximum sentence length in train
        for sentence in train_sentences:
            sentence_lengths.append(len(sentence))
        # end for

        slarr = np.asarray(sentence_lengths, dtype=np.float32)
        sl_max = np.max(slarr, keepdims=False)

        print(stack()[0][3] + ": maximum sentence length in training set is {0:.2f}".format(sl_max), file=sys.stderr, flush=True)

        sl_mean = np.mean(slarr)
        sl_stddev = np.std(slarr)

        if self._maxseqlen == 0:
            self._maxseqlen = int(sl_mean)

            print(stack()[0][3] + ": determined {0!s} as the maximum sentence length (mean = {1:.5f}, stddev = {2:.5f})".format(
                self._maxseqlen, sl_mean, sl_stddev), file=sys.stderr, flush=True)
        # end if

        train_samples = self._build_samples(train_sentences)
        dev_samples = self._build_samples(dev_sentences)
        test_samples = self._build_samples(test_sentences)

        train_sentences.extend(sentences)
        return (train_sentences, train_samples, dev_sentences, dev_samples, test_sentences, test_samples)

    def _build_unicode_props(self, data_samples: list):
        for sample in data_samples:
            for parts in sample:
                word = parts[0]

                if word != _zero_word:
                    self._uniprops.add_unicode_props(word)
                # end if
            # end for
        # end for

    def _get_lexical_features_for_pos_tagging(self, word: str) -> np.ndarray:
        """Will get an np.array of lexical features for word,
        including the possible MSDs."""

        # 1. Casing features
        features1 = np.zeros(len(Lex._case_patterns), dtype=np.float32)

        if word != _zero_word:
            for i in range(len(Lex._case_patterns)):
                patt = Lex._case_patterns[i]

                if patt.match(word):
                    features1[i] = 1.0
                # end if
            # end for
        # end if

        # 2. MSD features for word: the vector of possible MSDs
        features2 = np.zeros(
            self._msd.get_input_vector_size(), dtype=np.float32)

        if word != _zero_word:
            if self._lexicon.is_lex_word(word, exact_match=True):
                for msd in self._lexicon.get_word_ambiguity_class(word, exact_match=True):
                    msd_v = self._msd.msd_input_vector(msd)
                    features2 += msd_v
                # end for
            elif self._lexicon.is_lex_word(word.lower(), exact_match=True):
                for msd in self._lexicon.get_word_ambiguity_class(word.lower(), exact_match=True):
                    msd_v = self._msd.msd_input_vector(msd)
                    features2 += msd_v
                # end for
            elif word in MSD.punct_msd_inventory:
                msd = MSD.punct_msd_inventory[word]
                msd_v = self._msd.msd_input_vector(msd)
                features2 += msd_v
            elif MSD.punct_patt.match(word) != None:
                msd_v = self._msd.msd_input_vector("Z")
                features2 += msd_v
            elif Lex.number_pattern.match(word):
                msd_v = self._msd.msd_input_vector("Mc-s-d")
                features2 += msd_v
            elif Lex.bullet_number_pattern.match(word):
                msd_v = self._msd.msd_input_vector("Mc-s-b")
                features2 += msd_v
            else:
                # Use a better solution here.
                affix_msds = self._romorphology.ambiguity_class(word)

                if affix_msds:
                    for msd in affix_msds:
                        msd_v = self._msd.msd_input_vector(msd)
                        features2 += msd_v
                    # end for
                else:
                    # Default to the lexicon-based one.
                    affix_msds = self._lexicon.get_unknown_ambiguity_class(
                        word)

                    if affix_msds:
                        for msd in affix_msds:
                            msd_v = self._msd.msd_input_vector(msd)
                            features2 += msd_v
                        # end for
                    # end if
                    else:
                        features2 = self._msd.get_x_input_vector()
                    # end if
                # end if
            # end if

            # 1.0 normalization, let context choose
            features2[features2 > 1.0] = 1.0
        # end if

        # 4. Concatenate 1 and 2
        return np.concatenate((features1, features2))

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
                    self._rofeatures.compute_sentence_wide_features(
                        current_sentence)
                    sentences.append(current_sentence)
                    current_sentence = []
                    continue
                # end if

                if len(parts) != 3:
                    print(stack()[0][3] + ": line {0!s} in file {1!s} is not well-formed!".format(
                        line_count, file), file=sys.stderr, flush=True)
                else:
                    current_sentence.append((parts[0], parts[2]))
                    self._datavocabulary.add(parts[0])
                # end if
            # end all lines
        # end with

        return sentences

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

    def _normalize_vocabulary(self):
        new_vocabulary = set()

        for word in self._datavocabulary:
            if Lex.sentence_case_pattern.match(word):
                if not word.lower() in self._datavocabulary:
                    new_vocabulary.add(word)
                else:
                    print(stack()[
                          0][3] + ": removed word '{0}' from vocabulary".format(word), file=sys.stderr, flush=True)
                #end if
            # end if
            else:
                new_vocabulary.add(word)
            # end if
        # end for

        self._datavocabulary = new_vocabulary


if __name__ == '__main__':
    # Use this module to train the sentence splitter.
    tk = RoTokenizer()
    ss = RoSentenceSplitter(tk)
    ss.load()
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

    tg.train(train_sentences=training,
            dev_sentences=development, test_sentences=testing)
