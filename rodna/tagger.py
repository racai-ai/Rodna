import sys
import os
import re
from math import isclose
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from inspect import stack
from tqdm import tqdm
import json
from . import _device
from utils.CharUni import CharUni
from utils.Lex import Lex
from utils.MSD import MSD
from utils.datafile import read_all_ext_files_from_dir
from .splitter import RoSentenceSplitter
from .tokenizer import RoTokenizer
from .features import RoFeatures
from .morphology import RoInflect
from .embeddings import RoWordEmbeddings, zero_word, \
    start_word, end_word
from .crfmodel import CRFModelDataset, CRFModel
from config import CLS_TAGGER_MODEL_FOLDER, \
    CRF_TAGGER_MODEL_FOLDER, TAGGER_UNICODE_PROPERTY_FILE


class CLSModel(nn.Module):
    """This model takes the input tensor, learns a bidirectional RNN
    MSD encoding scheme and then, a bidirectional RNN MSD classification scheme."""

    # RNN state size
    _conf_rnn_size_1 = 256
    _conf_rnn_size_2 = 128

    def __init__(self,
            ro_embeddings: RoWordEmbeddings,
            lex_input_vector_size: int,
            ctx_input_vector_size: int,
            msd_encoding_vector_size: int,
            output_msd_size: int,
            runtime: bool,
            drop_prob: float = 0.33
        ):
        super().__init__()
        self._layer_embed = nn.Embedding.from_pretrained(
            embeddings=ro_embeddings.get_embeddings_weights(runtime=runtime),
            freeze=False)
        self._layer_rnn_1 = nn.GRU(
            input_size=lex_input_vector_size + ro_embeddings.get_vector_length(),
            hidden_size=CLSModel._conf_rnn_size_1,
            batch_first=True,
            bidirectional=True
        )
        self._layer_linear_enc = nn.Linear(
            in_features=2 * CLSModel._conf_rnn_size_1,
            out_features=msd_encoding_vector_size
        )
        self._layer_rnn_2 = nn.GRU(
            input_size=msd_encoding_vector_size,
            hidden_size=CLSModel._conf_rnn_size_2,
            batch_first=True,
            bidirectional=True
        )
        self._layer_linear_cls = nn.Linear(
            in_features=2 * CLSModel._conf_rnn_size_2 + ctx_input_vector_size,
            out_features=output_msd_size
        )
        self._layer_drop = nn.Dropout(p=drop_prob)
        self._sigmoid = nn.Sigmoid()
        self._layer_logsoftmax = nn.LogSoftmax(dim=2)
        self.to(device=_device)

    def forward(self, x):
        x_lex, x_emb, x_ctx = x
        b_size = x_emb.shape[0]
        h_0 = torch.zeros(
            2, b_size,
            CLSModel._conf_rnn_size_1).to(device=_device)

        # MSD encoding
        o_emb = self._layer_embed(x_emb)
        # Concatenate along features dimension
        o_lex_emb_conc = torch.cat([x_lex, o_emb], dim=2)
        o_bd_rnn, h_n = self._layer_rnn_1(o_lex_emb_conc, h_0)
        o_drop = self._layer_drop(o_bd_rnn)
        o_msd_enc = self._layer_linear_enc(o_drop)
        o_msd_enc = self._sigmoid(o_msd_enc)
        # End MSD encoding

        # MSD classification
        h_0 = torch.zeros(
            2, b_size,
            CLSModel._conf_rnn_size_2).to(device=_device)
        o_drop = self._layer_drop(o_msd_enc)
        o_bd_rnn, h_n = self._layer_rnn_2(o_drop, h_0)
        o_drop = self._layer_drop(o_bd_rnn)
        o_drop = torch.cat([o_drop, x_ctx], dim=2)
        o_msd_cls = self._layer_linear_cls(o_drop)
        o_msd_cls = self._layer_logsoftmax(o_msd_cls)
        # End MSD classification

        return o_msd_enc, o_msd_cls


class RoPOSTagger(object):
    """This class will do MSD POS tagging for Romanian.
    It will train/test the DNN models and also, given a string of Romanian text,
    it will split it in sentences, POS tag each sentence and return the list."""

    # How many words in a window to consider when constructing a sample.
    # Set to 0 to estimate it as the average sentence length in the training set.
    _conf_maxseqlen = 50
    # How much (%) to retain from the train data as dev/test sets
    _conf_dev_percent = 0.1
    # No test, for now, look at values on dev
    _conf_test_percent = 0.0
    _conf_epochs_cls = 10
    _conf_epochs_crf = 5
    _conf_with_tiered_tagging = False
    # Can be one of the following (see RoPOSTagger._run_sentence()):
    # - 'add': add the probabilities for each MSD that was assigned at position i, in each rolling window
    # - 'max': only keep the MSD with the highest probability at position i, from each rolling window
    # - 'cnt': add 1 to each different MSD that was assigned at position i, in each rolling window (majority voting)
    _conf_run_strategy = 'cnt'
    _conf_model_file = 'model.pt'
    _conf_config_file = 'config.json'

    def __init__(self, lexicon: Lex,
            tokenizer:RoTokenizer, morphology: RoInflect,
            splitter: RoSentenceSplitter):
        """Takes loaded instances of various objects."""
        self._splitter = splitter
        self._tokenizer = tokenizer
        self._uniprops = CharUni()
        self._lexicon = lexicon
        self._msd = self._lexicon.get_msd_object()
        self._rofeatures = RoFeatures(self._lexicon)
        self._romorphology = morphology
        self._wordembeddings = RoWordEmbeddings(self._lexicon)
        self._datavocabulary = set()
        self._maxseqlen = RoPOSTagger._conf_maxseqlen

    def _save(self):
        self._uniprops.save_unicode_props(TAGGER_UNICODE_PROPERTY_FILE)

        def _save_all(folder: str, model: nn.Module, config: dict):
            torchmodelfile = os.path.join(folder, RoPOSTagger._conf_model_file)
            torch.save(model.state_dict(), torchmodelfile)
            conffile = os.path.join(folder, RoPOSTagger._conf_config_file)

            with open(conffile, mode='w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            # end with
        # end def

        _save_all(CLS_TAGGER_MODEL_FOLDER, self._cls_model, self._cls_config)
        _save_all(CRF_TAGGER_MODEL_FOLDER, self._crf_model, self._crf_config)

    def load(self):
        self._uniprops.load_unicode_props(TAGGER_UNICODE_PROPERTY_FILE)
        self._wordembeddings.load_ids()

        # Load the CLS model
        def _load_conf(folder: str) -> dict:
            conffile = os.path.join(folder, RoPOSTagger._conf_config_file)
            config = {}

            with open(conffile, mode='r', encoding='utf-8') as f:
                config = json.load(f)
            # end with

            return config
        # end def

        self._cls_config = _load_conf(CLS_TAGGER_MODEL_FOLDER)
        self._crf_config = _load_conf(CRF_TAGGER_MODEL_FOLDER)

        self._cls_model = self._build_cls_model(
            lex_input_vector_size=self._cls_config['lex_input_vector_size'],
            ctx_input_vector_size=self._cls_config['ctx_input_vector_size'],
            msd_encoding_vector_size=self._cls_config['msd_encoding_vector_size'],
            output_msd_size=self._cls_config['output_msd_size'],
            runtime=True,
            drop_prob=float(self._cls_config['drop_prob'])
        )
        self._crf_model = self._build_crf_model(
            lex_input_vector_size=self._crf_config['lex_input_vector_size'],
            ctx_input_vector_size=self._crf_config['ctx_input_vector_size'],
            runtime=True,
            drop_prob=float(self._crf_config['drop_prob'])
        )

        def _load_model(folder: str, module: nn.Module):
            torchmodelfile = os.path.join(folder, RoPOSTagger._conf_model_file)
            module.load_state_dict(torch.load(
                torchmodelfile, map_location=_device))
            # Put model into eval mode. It is only used for inferencing.
            module.eval()
        # end def

        _load_model(CLS_TAGGER_MODEL_FOLDER, self._cls_model)
        _load_model(CRF_TAGGER_MODEL_FOLDER, self._crf_model)

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
        def _generate_cls_numpy(ml_type: str, examples: list) -> tuple:
            print(stack()[0][3] + ": building ENC/CLS {0} tensors".format(ml_type),
                  file=sys.stderr, flush=True)

            (x_lex_m, x_emb, x_ctx, y_enc, y_cls) = self._build_cls_io_tensors(examples)

            print(stack()[0][3] + ": {0} MSD x_lex.shape is {1!s}".format(
                ml_type, x_lex_m.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} x_emb.shape is {1!s}".format(
                ml_type, x_emb.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} x_ctx.shape is {1!s}".format(
                ml_type, x_ctx.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} ENC y.shape is {1!s}".format(
                ml_type, y_enc.shape), file=sys.stderr, flush=True)
            print(stack()[0][3] + ": {0} CLS y.shape is {1!s}".format(
                ml_type, y_cls.shape), file=sys.stderr, flush=True)

            return (x_lex_m, x_emb, x_ctx, y_enc, y_cls)
        # end def

        (x_lex_msd_train, x_emb_train, x_ctx_train,
            y_train_enc, y_train_cls) = _generate_cls_numpy('train', train_examples)
        (x_lex_msd_dev, x_emb_dev, x_ctx_dev,
            y_dev_enc, y_dev_cls) = _generate_cls_numpy('dev', dev_examples)
        (x_lex_msd_test, x_emb_test, x_ctx_test,
            y_test_enc, y_test_cls) = _generate_cls_numpy('test', dev_examples)

        (crf_x_lex_train, crf_x_emb_train, crf_x_ctx_train,
         crf_y_train) = self._build_crf_io_tensors(train_sentences)
        (crf_x_lex_dev, crf_x_emb_dev, crf_x_ctx_dev,
         crf_y_dev) = self._build_crf_io_tensors(dev_sentences)
        (crf_x_lex_test, crf_x_emb_test, crf_x_ctx_test,
         crf_y_test) = self._build_crf_io_tensors(test_sentences)
        crf_lex_input_dim = crf_x_lex_train[0].shape[1]
        crf_ctx_input_dim = crf_x_ctx_train[0].shape[1]

        # 5 Save RoInflect cache file for faster startup next time
        self._romorphology.save_cache()

        # 6.1 Build the PyTorch CRF model
        self._crf_config = {
            'lex_input_vector_size': crf_lex_input_dim,
            'ctx_input_vector_size': crf_ctx_input_dim,
            'drop_prob': 0.25
        }
        self._crf_model = self._build_crf_model(
            crf_lex_input_dim,
            crf_ctx_input_dim,
            False,
            0.25
        )
        self._crf_optimizer = Adam(self._crf_model.parameters(), lr=1e-3)

        pt_dataset_train = CRFModelDataset(
            crf_x_lex_train, crf_x_emb_train, crf_x_ctx_train, crf_y_train)
        pt_dataset_dev = CRFModelDataset(
            crf_x_lex_dev, crf_x_emb_dev, crf_x_ctx_dev, crf_y_dev)
        pt_dataset_test = CRFModelDataset(
            crf_x_lex_test, crf_x_emb_test, crf_x_ctx_test, crf_y_test)

        train_dataloader = DataLoader(
            dataset=pt_dataset_train, batch_size=4, shuffle=True, collate_fn=self._crf_collate_fn)
        dev_dataloader = DataLoader(
            dataset=pt_dataset_dev, batch_size=1, shuffle=False, collate_fn=self._crf_collate_fn)
        test_dataloader = DataLoader(
            dataset=pt_dataset_test, batch_size=1, shuffle=False, collate_fn=self._crf_collate_fn)

        for ep in range(1, RoPOSTagger._conf_epochs_crf + 1):
            # Fit model for one epoch
            self._crf_model.train(True)
            self._do_one_crf_epoch(epoch=ep, dataloader=train_dataloader)

            # Test model
            # Put model into eval mode first
            self._crf_model.eval()
            self._test_crf_model(dataloader=dev_dataloader, ml_set='dev')
        # end for

        self._crf_model.eval()
        self._test_crf_model(dataloader=test_dataloader, ml_set='test')

        # 7.1 Build the PyTorch CLS model
        cls_lex_input_dim = x_lex_msd_train.shape[2]
        cls_ctx_input_dim = x_ctx_train.shape[2]
        encoding_dim = y_train_enc.shape[2]
        output_msd_dim = self._msd.get_output_vector_size()

        self._cls_model = self._build_cls_model(
            lex_input_vector_size=cls_lex_input_dim,
            ctx_input_vector_size=cls_ctx_input_dim,
            msd_encoding_vector_size=encoding_dim,
            output_msd_size=output_msd_dim,
            runtime=False,
            drop_prob=0.25
        )
        self._cls_config = {
            'lex_input_vector_size': cls_lex_input_dim,
            'ctx_input_vector_size': cls_ctx_input_dim,
            'msd_encoding_vector_size': encoding_dim,
            'output_msd_size': output_msd_dim,
            'drop_prob': 0.25
        }

        # MSD encoding loss
        self._loss_fn_enc = nn.BCELoss()
        # MSD classification loss
        self._loss_fn_cls = nn.NLLLoss()
        self._cls_optimizer = Adam(self._cls_model.parameters(), lr=1e-3)

        # 7.2 Create PyTorch tensors and DataLoaders
        def _generate_cls_dataloader(x_lex_msd, x_emb, x_ctx, y_enc, y_cls) -> DataLoader:
            x_lex_msd = torch.tensor(x_lex_msd).to(_device)
            x_emb = torch.tensor(x_emb, dtype=torch.long).to(_device)
            x_ctx = torch.tensor(x_ctx).to(_device)
            y_enc = torch.tensor(y_enc).to(_device)
            y_cls = torch.tensor(y_cls, dtype=torch.long).to(_device)
            pt_dataset = TensorDataset(
                x_lex_msd, x_emb, x_ctx, y_enc, y_cls)
            dataloader = DataLoader(
                dataset=pt_dataset, batch_size=8, shuffle=True)

            return dataloader
        # end def

        train_dataloader = _generate_cls_dataloader(
            x_lex_msd_train, x_emb_train, x_ctx_train, y_train_enc, y_train_cls)
        dev_dataloader =_generate_cls_dataloader(
            x_lex_msd_dev, x_emb_dev, x_ctx_dev, y_dev_enc, y_dev_cls)
        test_dataloader = _generate_cls_dataloader(
            x_lex_msd_test, x_emb_test, x_ctx_test, y_test_enc, y_test_cls)
        
        # 7.3 Train the CLS model and test it
        for ep in range(1, RoPOSTagger._conf_epochs_cls + 1):
            # Fit model for one epoch
            self._cls_model.train(True)
            self._do_one_cls_epoch(epoch=ep, dataloader=train_dataloader)

            # Test model
            # Put model into eval mode first
            self._cls_model.eval()
            self._test_cls_model(dataloader=dev_dataloader, ml_set='dev')
        # end for

        self._cls_model.eval()
        self._test_cls_model(dataloader=test_dataloader, ml_set='test')

        # 7.4 Test on given sets
        self.test_on_sentence_set(sentences=dev_sentences)

        # 8. Save all models
        self._save()

    def _crf_collate_fn(self, batch) -> tuple:
        """Builds a batch for the CRFModel, taking care that
        all sentence lengths are the same! Cuts the batch size
        if next tensor had a different length."""
        
        lex_batch = []
        emb_batch = []
        ctx_batch = []
        yct_batch = []

        for inp_lex, inp_emb, inp_ctx, tar_ctg in batch:
            crt_slen = inp_lex.shape[0]
            
            if lex_batch:
                prv_slen = lex_batch[-1].shape[1]

                if crt_slen != prv_slen:
                    break
                # end if
            # end if

            # Move all tensors to _device
            inp_lex = inp_lex.view(
                1, inp_lex.shape[0], inp_lex.shape[1]).to(device=_device)
            inp_emb = inp_emb.view(1, -1).to(device=_device)
            inp_ctx = inp_ctx.view(
                1, inp_ctx.shape[0], inp_ctx.shape[1]).to(device=_device)
            tar_ctg = tar_ctg.view(1, -1).to(device=_device)
            lex_batch.append(inp_lex)
            emb_batch.append(inp_emb)
            ctx_batch.append(inp_ctx)
            yct_batch.append(tar_ctg)
        # end for

        return \
            torch.cat(lex_batch, dim=0), torch.cat(emb_batch, dim=0), \
            torch.cat(ctx_batch, dim=0), torch.cat(yct_batch, dim=0)

    def _test_crf_model(self, dataloader: DataLoader, ml_set: str):
        """Tests the CRF model with the dev/test sets."""

        correct = 0
        existing = 0

        for inp_lex, inp_emb, inp_ctx, tar_ctg in tqdm(dataloader, desc=f'Eval'):
            scores, batch_labels_list = self._crf_model(
                x=(inp_lex, inp_emb, inp_ctx))
            
            predicted_labels = []
            # len(batch_labels_list) is the batch size, tar_ctg.shape[0]
            for ll in batch_labels_list:
                predicted_labels.append(
                    torch.tensor(ll, dtype=torch.int32).view(1, -1).to(device=_device))
            # end for

            predicted_labels = torch.cat(predicted_labels, dim=0).to(device=_device)
            correct_labels = \
                (predicted_labels == tar_ctg).to(dtype=torch.int32)
            
            correct += torch.sum(correct_labels).item()
            existing += predicted_labels.numel()
        # end for

        acc = correct / existing
        print(f'Acc = {acc:.5f} on {ml_set}', file=sys.stderr, flush=True)

    def _test_cls_model(self, dataloader: DataLoader, ml_set: str):
        """Tests the CLS model with the dev/test sets."""

        correct = 0
        existing = 0

        for inp_lex, inp_emb, inp_ctx, _, tar_cls in tqdm(dataloader, desc=f'Eval'):
            _, outputs_cls = self._cls_model(x=(inp_lex, inp_emb, inp_ctx))
            outputs_cls = torch.exp(outputs_cls)
            predicted_labels = torch.argmax(outputs_cls, dim=2)
            correct_labels = \
                (predicted_labels == tar_cls).to(dtype=torch.int32)
            
            correct += torch.sum(correct_labels).item()
            existing += predicted_labels.numel()
        # end for

        acc = correct / existing
        print(f'Acc = {acc:.5f} on {ml_set}', file=sys.stderr, flush=True)

    def _do_one_cls_epoch(self, epoch: int, dataloader: DataLoader):
        """One CLSModel epoch."""
        running_loss = 0.
        epoch_loss = []
        counter = 0

        for inp_lex, inp_emb, inp_ctx, tar_enc, tar_cls in tqdm(dataloader, desc=f'Epoch {epoch}'):
            counter += 1

            # Zero your gradients for every batch!
            self._cls_optimizer.zero_grad()

            # Make predictions for this batch
            out_enc, out_cls = self._cls_model(x=(inp_lex, inp_emb, inp_ctx))

            loss_enc = self._loss_fn_enc(out_enc, tar_enc)
            # Have to swap axes for NLLLoss function
            # Classes are on the second dimension, dim=1
            out_cls = torch.swapaxes(out_cls, 1, 2)
            loss_cls = self._loss_fn_cls(out_cls, tar_cls)
            loss = loss_enc + loss_cls
            loss.backward()

            # Adjust learning weights and learning rate schedule
            self._cls_optimizer.step()

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

    def _do_one_crf_epoch(self, epoch: int, dataloader: DataLoader):
        """One CRFModel epoch."""

        running_loss = 0.
        epoch_loss = []
        counter = 0

        for inp_lex, inp_emb, inp_ctx, tar_ctg in tqdm(dataloader, desc=f'Epoch {epoch}'):
            counter += 1

            # Zero your gradients for every batch!
            self._crf_optimizer.zero_grad()

            # Make predictions for this batch and compute loss
            loss = self._crf_model.neg_log_likelihood(
                x=(inp_lex, inp_emb, inp_ctx), batch_tags=tar_ctg)
            loss.backward()

            # Adjust learning weights and learning rate schedule
            self._crf_optimizer.step()

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

    def _build_cls_model(self,
            lex_input_vector_size: int,
            ctx_input_vector_size: int,
            msd_encoding_vector_size: int,
            output_msd_size: int,
            runtime: bool,
            drop_prob: float = 0.33) -> CLSModel:
        return CLSModel(
            self._wordembeddings,
            lex_input_vector_size, ctx_input_vector_size,
            msd_encoding_vector_size, output_msd_size,
            runtime, drop_prob)

    def _build_crf_model(self,
            lex_input_vector_size: int,
            ctx_input_vector_size: int,
            runtime: bool,
            drop_prob: float = 0.33) -> CRFModel:
        return CRFModel(
            self._wordembeddings,
            self._msd.get_ctag_inventory(),
            lex_input_vector_size, ctx_input_vector_size,
            runtime, drop_prob)

    def _build_crf_io_tensors(self, sentences: list, runtime: bool = False) -> tuple:
        """Builds the CRF model input/output tensors."""

        xlex_tensor = []
        xemb_tensor = []
        xctx_tensor = []
        y_tensor = []

        for i in range(len(sentences)):
            sentence = sentences[i]
            # We should have that assert len(sample) == tx

            if i > 0 and i % 500 == 0:
                print(stack()[0][3] + ": processed {0!s}/{1!s} sentences".format(
                    i, len(sentences)), file=sys.stderr, flush=True)
            # end if

            sentence_lex_features = []
            sentence_embeddings = []
            sentence_ctags = []
            sentence_context = []

            for j in range(len(sentence)):
                parts = sentence[j]
                word = parts[0]
                msd = parts[1]
                ctag = self._msd.msd_to_ctag(msd)

                if not self._msd.is_valid_ctag(ctag):
                    print(stack()[0][3] + ": unknown CTAG [{0}] for MSD [{1}]".format(
                        ctag, msd), file=sys.stderr, flush=True)
                # end if

                feats = parts[2]
                tlabel = self._tokenizer.tag_word(word)

                if not runtime:
                    y_out_idx = self._msd.ctag_to_idx(ctag)
                else:
                    y_out_idx = self._msd.ctag_to_idx('X')
                # end if

                label_features = self._tokenizer.get_label_features(tlabel)
                uni_features = self._uniprops.get_unicode_features(word)
                lexical_features_ctag = self._get_lexical_features_for_pos_tagging(
                    word, with_ctags=True)

                # This is the featurized version of a word in the sequence
                x_lex_ctag = np.concatenate(
                    (label_features, uni_features, lexical_features_ctag))

                if word == zero_word:
                    x_lex_ctag = np.zeros(x_lex_ctag.shape, dtype=np.float32)
                # end if

                # Computing id for word
                x_wid = self._wordembeddings.get_word_id(word)
                # Computing external features for word
                x_ctx = self._rofeatures.get_context_feature_vector(feats)

                sentence_lex_features.append(torch.tensor(
                    x_lex_ctag, dtype=torch.float32).view(1, -1))
                sentence_embeddings.append(x_wid)
                sentence_context.append(torch.tensor(
                    x_ctx, dtype=torch.float32).view(1, -1))
                sentence_ctags.append(y_out_idx)
            # end j

            y_tensor.append(torch.tensor(sentence_ctags, dtype=torch.int32))
            xemb_tensor.append(torch.tensor(
                sentence_embeddings, dtype=torch.long))
            xlex_tensor.append(torch.cat(sentence_lex_features, dim=0))
            xctx_tensor.append(torch.cat(sentence_context, dim=0))
        # end i

        return (xlex_tensor, xemb_tensor, xctx_tensor, y_tensor)

    def _build_cls_io_tensors(self, data_samples, runtime: bool = False) -> tuple:
        # No of examples
        m = len(data_samples)
        # This should be equal to self._maxseqlen
        tx = len(data_samples[0])
        assert tx == self._maxseqlen

        ### Inputs
        # Lexical tensor
        xlex_msd_tensor = None
        # Word embeddings tensor
        xemb_tensor = np.empty((m, tx), dtype=np.int32)
        # Externally computed contextual features
        xctx_tensor = np.empty(
            (m, tx, len(RoFeatures.romanian_pos_tagging_features)), dtype=np.float32)

        ### Ground truth outputs
        # Ys for the MSD encoding part
        y_tensor_enc = np.empty(
            (m, tx, self._msd.get_input_vector_size()), dtype=np.float32)
        # Ys for the MSD classification part
        y_tensor_cls = np.empty((m, tx), dtype=np.int32)

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
                    y_out = self._msd.msd_to_idx(msd)
                else:
                    y_in = self._msd.get_x_input_vector()
                    y_out = self._msd.get_x_idx()
                # end if

                label_features = self._tokenizer.get_label_features(tlabel)
                uni_features = self._uniprops.get_unicode_features(word)
                lexical_features_msd = self._get_lexical_features_for_pos_tagging(
                    word, with_ctags=False)

                # This is the featurized version of a word in the sequence
                x_lex_msd = np.concatenate(
                    (label_features, uni_features, lexical_features_msd))

                if word == zero_word:
                    x_lex_msd = np.zeros(x_lex_msd.shape, dtype=np.float32)
                # end if

                if xlex_msd_tensor is None:
                    xlex_msd_tensor = np.empty(
                        (m, tx, x_lex_msd.shape[0]), dtype=np.float32)
                # end if

                # Computing id for word
                x_wid = self._wordembeddings.get_word_id(word)
                # Computing external features for word
                x_ctx = self._rofeatures.get_context_feature_vector(feats)

                xlex_msd_tensor[i, j, :] = x_lex_msd
                xemb_tensor[i, j] = x_wid
                xctx_tensor[i, j, :] = x_ctx
                y_tensor_enc[i, j, :] = y_in
                y_tensor_cls[i, j] = y_out
            # end j
        # end i

        return (xlex_msd_tensor, xemb_tensor, xctx_tensor, y_tensor_enc, y_tensor_cls)

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
            padded_sentence.append((zero_word, 'X', []))
        # end with

        return padded_sentence

    def _add_sentence_boundaries(self, sentence: list):
        """Add sentences START/END words for CRF modeling."""
        if sentence[0][0] != start_word:
            sentence.insert(0,
                (start_word, self._msd.get_start_end_tags('start', ctag=False), []))
        # end if

        if sentence[-1][0] != end_word:
            sentence.append(
                (end_word, self._msd.get_start_end_tags('end', ctag=False), []))
        # end if

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

        return (train_sentences, train_samples, dev_sentences, dev_samples, test_sentences, test_samples)

    def _build_unicode_props(self, data_samples: list):
        for sample in data_samples:
            for parts in sample:
                word = parts[0]

                if word != zero_word:
                    self._uniprops.add_unicode_props(word)
                # end if
            # end for
        # end for

    def _get_lexical_features_for_pos_tagging(self, word: str, with_ctags: bool) -> np.ndarray:
        """Will get an np.array of lexical features for word,
        including the possible MSDs."""

        # 1. Casing features
        features1 = np.zeros(len(Lex._case_patterns), dtype=np.float32)

        if word != zero_word:
            for i in range(len(Lex._case_patterns)):
                patt = Lex._case_patterns[i]

                if patt.match(word):
                    features1[i] = 1.0
                # end if
            # end for
        # end if

        # 2. MSD features for word: the vector of possible MSDs
        # If working with CTAGs, this is the vector for possible CTAGs
        if with_ctags:
            features2 = np.zeros(
                self._msd.get_ctag_input_vector_size(), dtype=np.float32)
        else:
            features2 = np.zeros(
                self._msd.get_input_vector_size(), dtype=np.float32)
        # end if

        def _pos_label_vector(msd: str) -> np.ndarray:
            if with_ctags:
                ctag = self._msd.msd_to_ctag(msd)
                lbl_v = self._msd.ctag_input_vector(ctag)
            else:
                lbl_v = self._msd.msd_input_vector(msd)
            # end if

            return lbl_v
        # end def

        if word != zero_word:
            if self._lexicon.is_lex_word(word, exact_match=True):
                for msd in self._lexicon.get_word_ambiguity_class(word, exact_match=True):
                    features2 += _pos_label_vector(msd)
                # end for
            elif self._lexicon.is_lex_word(word.lower(), exact_match=True):
                for msd in self._lexicon.get_word_ambiguity_class(word.lower(), exact_match=True):
                    features2 += _pos_label_vector(msd)
                # end for
            elif word in MSD.punct_msd_inventory:
                msd = MSD.punct_msd_inventory[word]
                features2 += _pos_label_vector(msd)
            elif MSD.punct_patt.match(word) != None:
                features2 += _pos_label_vector("Z")
            elif Lex.number_pattern.match(word):
                features2 += _pos_label_vector("Mc-s-d")
            elif Lex.bullet_number_pattern.match(word):
                features2 += _pos_label_vector("Mc-s-b")
            elif word == start_word:
                features2 += _pos_label_vector("Lb")
            elif word == end_word:
                features2 += _pos_label_vector("Le")
            else:
                # Use a better solution here.
                affix_msds = self._romorphology.ambiguity_class(word)

                if affix_msds:
                    for msd in affix_msds:
                        features2 += _pos_label_vector(msd)
                    # end for
                else:
                    # Default to the lexicon-based one.
                    affix_msds = self._lexicon.get_unknown_ambiguity_class(
                        word)

                    if affix_msds:
                        for msd in affix_msds:
                            features2 += _pos_label_vector(msd)
                        # end for
                    # end if
                    else:
                        if with_ctags:
                            features2 = self._msd.get_ctag_x_input_vector()
                        else:
                            features2 = self._msd.get_x_input_vector()
                        # end if
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
                    # No sentence boundaries for CRF modeling.
                    # The layer adds START/END tags for its internal purposes.
                    #self._add_sentence_boundaries(current_sentence)
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

    def _normalize_vocabulary(self):
        new_vocabulary = set()

        for word in self._datavocabulary:
            if Lex.sentence_case_pattern.match(word):
                if word.lower() not in self._datavocabulary:
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

    def _get_predicted_msd(self, y_pred: np.ndarray, epsilon: float = 2e-5) -> list:
        """Did not take into account that we could have multiple MSDs with maximum
        probabilities, so account for that."""
        result = []

        best_idx = np.argmax(y_pred)
        msd_p = y_pred[best_idx]
        msd = self._msd.idx_to_msd(best_idx)
        result.append((msd, msd_p))

        a = y_pred
        b = np.array([msd_p] * y_pred.shape[0], dtype=np.float32)
        d = abs(a - b)
        best_idxes = np.argwhere(d < epsilon)
        best_idxes = np.reshape(best_idxes, (best_idxes.shape[0],))

        for bi in best_idxes:
            p = y_pred[bi]
            m = self._msd.idx_to_msd(bi)

            if m != msd:
                result.append((m, p))
            # end if
        # end for

        return result

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

        # 1. Get the model predicted MSDs
        msd_predictions = self._get_predicted_msd(y_pred)
        best_pred_msds = [m for m, _ in msd_predictions]

        # 2. Get the extended word ambiguity class, as the predicted MSD may
        # be outside this set.
        known_word_msds = set()

        if self._lexicon.is_lex_word(word):
            known_word_msds.update(
                self._lexicon.get_word_ambiguity_class(word))
        # end if

        # 3. Get extended ambiguity classes only for content words
        can_be_content = False

        for msd in best_pred_msds + list(known_word_msds):
            if self._lexicon.content_word_pos_pattern.match(msd):
                can_be_content = True
                break
            # end if
        # end for

        if can_be_content:
            aclass = self._romorphology.ambiguity_class(word)

            if not aclass:
                aclass = self._lexicon.get_unknown_ambiguity_class(word)
            # end if

            known_word_msds.update(aclass)
        # end if

        # This list cannot be empty!
        computed_word_msds = best_pred_msds

        if known_word_msds:
            computed_word_msds = list(
                known_word_msds.intersection(best_pred_msds))
        # end if

        if computed_word_msds:
            # 3. Model predicted MSD is in the extended ambiguity class. Hurray!
            best_pred_msd = ''
            best_pred_msd_p = 0.

            for m1 in computed_word_msds:
                for m2, p in msd_predictions:
                    if m1 == m2 and best_pred_msd_p < p:
                        best_pred_msd_p = p
                        best_pred_msd = m1
                    # end if
                # end for
            # end for

            return (best_pred_msd, best_pred_msd_p)
        elif known_word_msds:
            computed_word_msds = list(known_word_msds)
        else:
            computed_word_msds = best_pred_msds
        # end if

        # 4. Model predicted MSD is not in the extended ambiguity class.
        # Choose the highest probability MSD from the computed MSD list.
        best_acls_msd_p = 0.
        best_acls_msd = ''

        for msd in computed_word_msds:
            idx = self._msd.msd_to_idx(msd)
            msd_p = y_pred[idx]

            if msd_p > best_acls_msd_p:
                best_acls_msd_p = msd_p
                best_acls_msd = msd
            # end if
        # end for extended ambiguity class

        print(stack()[0][3] +
              ": word '{0}' -> got suboptimal MSD {1}/{2:.2f}"
              .format(word, best_acls_msd, best_acls_msd_p), file=sys.stderr, flush=True)

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

        both_aclass = set(lex_aclass).union(infl_aclass)
        result_aclass = list(both_aclass.intersection(mclass))

        if result_aclass:
            print(
                f"_tiered_tagging[standard]: word '{word}' got MSDs [{', '.join(result_aclass)}] for CTAG [{ctag}]",
                file=sys.stderr, flush=True)
            return result_aclass
        # end if

        amend_aclass = self._lexicon.amend_ambiguity_class(word, both_aclass)
        result_aclass = list(amend_aclass.intersection(mclass))

        if result_aclass:
            print(
                f"_tiered_tagging[extended]: word '{word}' got MSDs [{', '.join(result_aclass)}] for CTAG [{ctag}]",
                file=sys.stderr, flush=True)
            return result_aclass
        else:
            # If result_aclass is empty, tiered tagging failed:
            # CTAG is very different than ambiguity classes!
            list_amend_aclass = list(amend_aclass)
            print(
                f"_tiered_tagging[error]: word '{word}' got MSDs [{', '.join(list_amend_aclass)}] for CTAG [{ctag}]",
                file=sys.stderr, flush=True)
            return list_amend_aclass
        # end if

    def tag_sentence(self, sentence: list) -> list:
        """Main method of this class. Takes a list of tokens from the Romanian
        tokenizer and outputs the MSD-tagged list of words, each MSD with its
        confidence score."""

        internal_sentence = [
            (w, 'X') for w, tt in sentence \
                if tt != 'SPACE' and tt != 'EOL' and \
                    not re.fullmatch('^\\s*$', w)
        ]
        self._rofeatures.compute_sentence_wide_features(internal_sentence)

        return self._eval_sentence(internal_sentence)

    def _eval_sentence(self, sentence, tiered_tagging: bool = True, strategy: str = 'cnt', debug_fh = None):
        """Strategy can be `'add'` or `'max'`. When `'add'`ing contributions from the LM and CRF layers,
        a dict of predicted MSDs is kept at each position in the sentence and we add the probabilities
        coming from each rolling window, at position `i`. The best MSD wins at position `i`.

        We can also keep the winning, predicted MSD (by its `'max'` probability) at each position `i` in the sentence,
        with each rolling window.
        
        When `debug_fh is not None`, the `runtime is False` for the tensor computation routines."""
        runtime_flag = True

        if debug_fh:
            runtime_flag = False
        # end if

        # 1. Build the fixed-length samples from the input sentence
        sent_samples = self._build_samples([sentence])

        # 2.1 Get the input tensors for CLS
        # y_pred_cls is the POS tagger sequence with RNNs, one-hot
        (cls_x_lex, cls_x_emb, cls_x_ctx, y_enc, y_cls) = \
            self._build_cls_io_tensors(sent_samples, runtime=runtime_flag)
        cls_x_lex = torch.tensor(cls_x_lex).to(_device)
        cls_x_emb = torch.tensor(cls_x_emb, dtype=torch.long).to(_device)
        cls_x_ctx = torch.tensor(cls_x_ctx).to(_device)
        y_enc = torch.tensor(y_enc).to(_device)
        y_cls = torch.tensor(y_cls, dtype=torch.long).to(_device)
        out_enc, out_cls = self._cls_model(x=(cls_x_lex, cls_x_emb, cls_x_ctx))
        y_pred_cls = torch.exp(out_cls).cpu().detach().numpy()

        # This is how many samples in the sentence
        assert y_pred_cls.shape[0] == len(sent_samples)
        # This is the configured length of the sequence
        assert y_pred_cls.shape[1] == len(sent_samples[0])

        viterbi_sequence = []

        if tiered_tagging:
            # 2.2 Get the input tensors for CRF
            # viterbi_sequence is the POS tagger sequence with CRFs, CTAG index
            (crf_x_lex_l, crf_x_emb_l, crf_x_ctx_l, y_crf_l) = \
                self._build_crf_io_tensors([sentence], runtime=runtime_flag)
            # We only have 1 sentence
            crf_x_lex = crf_x_lex_l[0]
            crf_x_emb = crf_x_emb_l[0]
            crf_x_ctx = crf_x_ctx_l[0]
            y_crf = y_crf_l[0]
            crf_x_lex = crf_x_lex.view(
                1, crf_x_lex.shape[0], crf_x_lex.shape[1]).to(device=_device)
            crf_x_emb = crf_x_emb.view(1, -1).to(device=_device)
            crf_x_ctx = crf_x_ctx.view(
                1, crf_x_ctx.shape[0], crf_x_ctx.shape[1]).to(device=_device)
            y_crf = y_crf.view(1, -1).to(device=_device)
            scores, batch_labels_list = self._crf_model(
                x=(crf_x_lex, crf_x_emb, crf_x_ctx))
            viterbi_sequence = batch_labels_list[0]
            assert len(viterbi_sequence) == len(sentence)
        # end if

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
                        y2 = viterbi_sequence[i + j]
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

                if tiered_tagging:
                    if crf_correct == 3:
                        print("CRF=", file=debug_fh, flush=True)
                    elif crf_correct > 0:
                        print("CRF~ c={0};pm={1};mx={2}".format(
                            ','.join(crf_msd_cnt), ','.join(crf_msd_pmass), ','.join(crf_msd_maxp)), file=debug_fh, flush=True)
                    else:
                        print("CRF! c={0};pm={1};mx={2}".format(
                            ','.join(crf_msd_cnt), ','.join(crf_msd_pmass), ','.join(crf_msd_maxp)), file=debug_fh, flush=True)
                    # end if
                else:
                    print("", file=debug_fh, flush=True)
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
            print(file=debug_fh, flush=True)
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

    def test_on_sentence_set(self, sentences):
        all_count = 0
        correct_count = 0

        with open('tagger-debug.txt', mode='w', encoding='utf-8') as f:
            for sentence in tqdm(sentences, desc=f'Eval'):
                tagged_sentence = tag._eval_sentence(
                    sentence,
                    tiered_tagging=RoPOSTagger._conf_with_tiered_tagging,
                    strategy=RoPOSTagger._conf_run_strategy,
                    debug_fh=f)
        
                for i in range(len(sentence)):
                    if tagged_sentence[i][1] == sentence[i][1]:
                        correct_count += 1
                    # end if
                # end for

                all_count += len(sentence)
            # end all sentences
        # end with

        acc = correct_count / all_count
        print(f'Acc = {acc:.5f}')
        self._romorphology.save_cache()


if __name__ == '__main__':
    # Use this module to train the sentence splitter.
    lex = Lex()
    tok = RoTokenizer(lex)
    spl = RoSentenceSplitter(lex, tok)
    spl.load()
    mor = RoInflect(lex)
    mor.load()
    tag = RoPOSTagger(lex, tok, mor, spl)

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
    training = tag.read_tagged_file(training_file)

    development_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-dev.tab")
    print(stack()[0][3] + ": reading development file {0!s}".format(
        development_file), file=sys.stderr, flush=True)
    development = tag.read_tagged_file(development_file)

    testing_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-test.tab")
    print(stack()[0][3] + ": reading testing file {0!s}".format(
        testing_file), file=sys.stderr, flush=True)
    testing = tag.read_tagged_file(testing_file)

    tag.train(train_sentences=training,
            dev_sentences=development, test_sentences=testing)
