import sys
import os
import re
from typing import List, Tuple, Dict, Set
from math import isclose
from random import shuffle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
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
from .crfmodel import CRFModel
from .clsmodel import CLSModel
from . import CLS_TAGGER_MODEL_FOLDER, \
    CRF_TAGGER_MODEL_FOLDER, TAGGER_UNICODE_PROPERTY_FILE, \
    TAGGER_MODEL_FOLDER, BERT_FOR_CLS_TAGGER_FOLDER, \
    BERT_FOR_CRF_TAGGER_FOLDER, logger, logging, log_once
from .bert_model import RoBERTModel, \
    zero_word, start_word, end_word, dumitrescu_bert_v1


class TaggerDataset(Dataset):
    """This is a dataset for both the CLS and the CRF models."""

    def __init__(self, sentences: List[List[Tuple[str, str, List[str]]]]):
        super().__init__()
        self._data = []
        self._fill_data_by_len(sentences=sentences)
        self.reshuffle()

    def _fill_data_by_len(self, sentences: List[List[Tuple[str, str, List[str]]]]):
        # Group tensors by sentence length, for batch processing.
        self._data_by_len: Dict[int, List[List[Tuple]]] = {}

        for snt in sentences:
            snt_len = len(snt)

            if snt_len not in self._data_by_len:
                self._data_by_len[snt_len] = []
            # end if

            self._data_by_len[snt_len].append(snt)
        # end for

        # Sort sentences by length from high to low
        self._sentence_lengths = list(self._data_by_len.keys())
        self._sentence_lengths.sort(reverse=True)

    def reshuffle(self):
        self._data.clear()

        for snt_len in self._sentence_lengths:
            shuffle(self._data_by_len[snt_len])

            for snt in self._data_by_len[snt_len]:
                self._data.append(snt)
            # end for
        # end for

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class RoPOSTagger(object):
    """This class will do MSD POS tagging for Romanian.
    It will train/test the DNN models and also, given a string of Romanian text,
    it will split it in sentences, POS tag each sentence and return the list."""

    # How much (%) to retain from the train data as dev/test sets
    _conf_dev_percent = 0.1
    # No test, for now, look at values on dev
    _conf_test_percent = 0.1
    _conf_epochs_cls = 10
    _conf_epochs_crf = 2
    _conf_with_tiered_tagging = True
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
        self._cls_ro_bert.save(destination_folder=BERT_FOR_CLS_TAGGER_FOLDER)
        _save_all(CRF_TAGGER_MODEL_FOLDER, self._crf_model, self._crf_config)
        self._crf_ro_bert.save(destination_folder=BERT_FOR_CRF_TAGGER_FOLDER)

    def load(self):
        def _load_conf(folder: str) -> dict:
            conffile = os.path.join(folder, RoPOSTagger._conf_config_file)
            config = {}

            with open(conffile, mode='r', encoding='utf-8') as f:
                config = json.load(f)
            # end with

            return config
        # end def

        def _load_model(folder: str, module: nn.Module):
            torchmodelfile = os.path.join(folder, RoPOSTagger._conf_model_file)
            module.load_state_dict(torch.load(
                torchmodelfile, map_location=_device))
            # Put model into eval mode. It is only used for inferencing.
            module.eval()
        # end def

        self._uniprops.load_unicode_props(TAGGER_UNICODE_PROPERTY_FILE)

        self._cls_config = _load_conf(CLS_TAGGER_MODEL_FOLDER)
        self._cls_ro_bert = RoBERTModel(
            path_or_name=BERT_FOR_CLS_TAGGER_FOLDER)
        self._cls_model = self._build_cls_model(
            emb_input_vector_size=self._cls_ro_bert.get_embedding_size(),
            lex_input_vector_size=self._cls_config['lex_input_vector_size'],
            ctx_input_vector_size=self._cls_config['ctx_input_vector_size'],
            msd_encoding_vector_size=self._cls_config['msd_encoding_vector_size'],
            output_msd_size=self._cls_config['output_msd_size'],
            drop_prob=float(self._cls_config['drop_prob'])
        )
        _load_model(CLS_TAGGER_MODEL_FOLDER, self._cls_model)

        self._crf_config = _load_conf(CRF_TAGGER_MODEL_FOLDER)
        self._crf_ro_bert = RoBERTModel(
            path_or_name=BERT_FOR_CRF_TAGGER_FOLDER)
        self._crf_model = self._build_crf_model(
            emb_input_vector_size=self._crf_ro_bert.get_embedding_size(),
            lex_input_vector_size=self._crf_config['lex_input_vector_size'],
            ctx_input_vector_size=self._crf_config['ctx_input_vector_size'],
            drop_prob=float(self._crf_config['drop_prob'])
        )
        _load_model(CRF_TAGGER_MODEL_FOLDER, self._crf_model)

    def train(self,
              train_sentences: List[List[Tuple[str, str, List[str]]]],
              dev_sentences: List[List[Tuple[str, str, List[str]]]],
              test_sentences: List[List[Tuple[str, str, List[str]]]]):
        logger.info(f"Got train set with [{len(train_sentences)}] sentences")
        logger.info(f"Got dev set with [{len(dev_sentences)}] sentences")
        logger.info(f"Got test set with [{len(test_sentences)}] sentences")

        # 3. Build the Unicode properties on the train set
        self._build_unicode_props(train_sentences)

        # 4 Load BERT model for fine-tuning
        self._crf_ro_bert = RoBERTModel(path_or_name=dumitrescu_bert_v1, fine_tune=True)
        crf_emb_input_dim = self._crf_ro_bert.get_embedding_size()

        # 5. Build the PyTorch CRF model
        crf_x_lex_0, _, crf_x_ctx_0, _ = \
            self._build_crf_io_tensor(sentence=train_sentences[0])
        crf_lex_input_dim = crf_x_lex_0.shape[1]
        crf_ctx_input_dim = crf_x_ctx_0.shape[1]

        self._crf_config = {
            'lex_input_vector_size': crf_lex_input_dim,
            'ctx_input_vector_size': crf_ctx_input_dim,
            'drop_prob': 0.25
        }
        self._crf_model = self._build_crf_model(
            crf_emb_input_dim,
            crf_lex_input_dim,
            crf_ctx_input_dim,
            0.25
        )

        # Don’t decay LayerNorm & biases
        bert_decay_params, bert_no_decay_params = [], []

        for name, param in self._crf_ro_bert.bert_model.named_parameters():
            if param.ndim == 1 or name.endswith(".bias"):
                bert_no_decay_params.append(param)
            else:
                bert_decay_params.append(param)
            # end if
        # end for

        self._crf_optimizer = AdamW([
            {"params": bert_decay_params, "weight_decay": 0.01, "lr": 2e-5},
            {"params": bert_no_decay_params, "weight_decay": 0.0, "lr": 2e-5},
            {"params": self._crf_model.parameters(), "weight_decay": 0.0, "lr": 1e-3}]
        )
        self._crf_lr_scheduler = ExponentialLR(optimizer=self._crf_optimizer, gamma=0.95)

        dataset_train = TaggerDataset(sentences=train_sentences)
        dataset_dev = TaggerDataset(sentences=dev_sentences)
        dataset_test = TaggerDataset(sentences=test_sentences)

        train_dataloader = DataLoader(
            dataset=dataset_train, batch_size=4, shuffle=False, collate_fn=self._crf_collate_fn)
        # batch_size=1 is mandatory so that the evaluation function
        # is considering all examples correctly.
        dev_dataloader = DataLoader(
            dataset=dataset_dev, batch_size=1, shuffle=False, collate_fn=self._crf_collate_fn)
        test_dataloader = DataLoader(
            dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=self._crf_collate_fn)

        for ep in range(1, RoPOSTagger._conf_epochs_crf + 1):
            # Fit model for one epoch
            self._crf_model.train()
            self._do_one_crf_epoch(epoch=ep, dataloader=train_dataloader)

            # Test model
            # Put model into eval mode first
            self._crf_model.eval()
            self._test_crf_model(dataloader=dev_dataloader, ml_set='dev')

            self._crf_lr_scheduler.step()
            bert_lr, _, crf_lr = self._crf_lr_scheduler.get_last_lr()
            logger.info(f'Setting new BERT LR to [{bert_lr:.7f}]')
            logger.info(f'Setting new CRF LR to [{crf_lr:.7f}]')

            # Reshuffle the train set
            dataset_train.reshuffle()
        # end for

        self._crf_model.eval()
        self._test_crf_model(dataloader=test_dataloader, ml_set='test')

        # 7.1 Build the PyTorch CLS model
        self._cls_ro_bert = RoBERTModel(path_or_name=dumitrescu_bert_v1, fine_tune=True)
        cls_emb_input_dim = self._cls_ro_bert.get_embedding_size()

        cls_x_lex_0, _, cls_x_ctx_0, cls_x_enc_0, _ = \
            self._build_cls_io_tensor(sample=train_sentences[0])
        cls_lex_input_dim = cls_x_lex_0.shape[1]
        cls_ctx_input_dim = cls_x_ctx_0.shape[1]
        cls_enc_input_dim = cls_x_enc_0.shape[1]
        output_msd_dim = self._msd.get_output_vector_size()

        self._cls_model = self._build_cls_model(
            emb_input_vector_size=cls_emb_input_dim,
            lex_input_vector_size=cls_lex_input_dim,
            ctx_input_vector_size=cls_ctx_input_dim,
            msd_encoding_vector_size=cls_enc_input_dim,
            output_msd_size=output_msd_dim,
            drop_prob=0.25
        )
        self._cls_config = {
            'lex_input_vector_size': cls_lex_input_dim,
            'ctx_input_vector_size': cls_ctx_input_dim,
            'msd_encoding_vector_size': cls_enc_input_dim,
            'output_msd_size': output_msd_dim,
            'drop_prob': 0.25
        }

        # MSD encoding loss
        self._loss_fn_enc = nn.BCELoss()
        # MSD classification loss
        self._loss_fn_cls = nn.NLLLoss()

        # Don’t decay LayerNorm & biases
        bert_decay_params, bert_no_decay_params = [], []

        for name, param in self._cls_ro_bert.bert_model.named_parameters():
            if param.ndim == 1 or name.endswith(".bias"):
                bert_no_decay_params.append(param)
            else:
                bert_decay_params.append(param)
            # end if
        # end for

        self._cls_optimizer = AdamW([
            {"params": bert_decay_params, "weight_decay": 0.01, "lr": 2e-5},
            {"params": bert_no_decay_params, "weight_decay": 0.0, "lr": 2e-5},
            {"params": self._cls_model.parameters(), "weight_decay": 0.0, "lr": 1e-3}]
        )
        self._cls_lr_scheduler = ExponentialLR(optimizer=self._cls_optimizer, gamma=0.95)

        train_dataloader = DataLoader(
            dataset=dataset_train, batch_size=4, shuffle=False, collate_fn=self._cls_collate_fn)
        # batch_size=1 is mandatory so that the evaluation function
        # is considering all examples correctly.
        dev_dataloader = DataLoader(
            dataset=dataset_dev, batch_size=1, shuffle=False, collate_fn=self._cls_collate_fn)
        test_dataloader = DataLoader(
            dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=self._cls_collate_fn)
        
        # 7.3 Train the CLS model and test it
        for ep in range(1, RoPOSTagger._conf_epochs_cls + 1):
            # Fit model for one epoch
            self._cls_model.train()
            self._do_one_cls_epoch(epoch=ep, dataloader=train_dataloader)

            # Test model
            # Put model into eval mode first
            self._cls_model.eval()
            self._test_cls_model(dataloader=dev_dataloader, ml_set='dev')

            self._cls_lr_scheduler.step()
            bert_lr, _, cls_lr = self._cls_lr_scheduler.get_last_lr()
            logger.info(f'Setting new BERT LR to [{bert_lr:.7f}]')
            logger.info(f'Setting new CRF LR to [{cls_lr:.7f}]')

            # Reshuffle the train set
            dataset_train.reshuffle()
        # end for

        self._cls_model.eval()
        self._test_cls_model(dataloader=test_dataloader, ml_set='test')

        # 7.4 Save all models
        self._save()
        # 5. Save RoInflect cache file for faster startup next time
        self._romorphology.save_cache()

    def _trim_batch(self, batch: List[List[Tuple[str, str, List[str]]]]) -> List[List[Tuple[str, str, List[str]]]]:
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

    def _crf_collate_fn(self, batch: List[List[Tuple[str, str, List[str]]]]) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
        """Builds a batch taking care that all sentence lengths are the same!
        Cuts the batch size if next tensor had a different length."""

        batch = self._trim_batch(batch=batch)
        
        lex_batch = []
        emb_batch = []
        ctx_batch = []
        yct_batch = []

        for sentence in batch:
            inp_lex, inp_emb, inp_ctx, tar_ctg = \
                self._build_crf_io_tensor(sentence=sentence)

            # Move all tensors to _device
            inp_lex = inp_lex.view(1, inp_lex.shape[0], inp_lex.shape[1])
            inp_emb = inp_emb.view(1, inp_emb.shape[0], inp_emb.shape[1])
            inp_ctx = inp_ctx.view(1, inp_ctx.shape[0], inp_ctx.shape[1])
            tar_ctg = tar_ctg.view(1, -1)
            
            lex_batch.append(inp_lex)
            emb_batch.append(inp_emb)
            ctx_batch.append(inp_ctx)
            yct_batch.append(tar_ctg)
        # end for

        return \
            torch.cat(lex_batch, dim=0), torch.cat(emb_batch, dim=0), \
            torch.cat(ctx_batch, dim=0), torch.cat(yct_batch, dim=0)
    
    def _cls_collate_fn(self, batch: List[List[Tuple[str, str, List[str]]]]) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                      torch.Tensor, torch.Tensor,
                                                                                      torch.Tensor]:
        batch = self._trim_batch(batch=batch)

        lex_batch = []
        emb_batch = []
        ctx_batch = []
        enc_batch = []
        ycls_batch = []
        
        # We know all samples have the same length.
        # That is, _conf_maxseqlen
        for sample in batch:
            inp_lex, inp_emb, inp_ctx, inp_enc, out_cls = \
                self._build_cls_io_tensor(sample=sample)

            # Move all tensors to _device
            inp_lex = inp_lex.view(1, inp_lex.shape[0], inp_lex.shape[1])
            inp_emb = inp_emb.view(1, inp_emb.shape[0], inp_emb.shape[1])
            inp_ctx = inp_ctx.view(1, inp_ctx.shape[0], inp_ctx.shape[1])
            inp_enc = inp_enc.view(1, inp_enc.shape[0], inp_enc.shape[1])
            out_cls = out_cls.view(1, -1)

            lex_batch.append(inp_lex)
            emb_batch.append(inp_emb)
            ctx_batch.append(inp_ctx)
            enc_batch.append(inp_enc)
            ycls_batch.append(out_cls)
        # end for

        return torch.cat(lex_batch, dim=0).to(_device), \
            torch.cat(emb_batch, dim=0).to(_device), \
            torch.cat(ctx_batch, dim=0).to(_device), \
            torch.cat(enc_batch, dim=0).to(_device), \
            torch.cat(ycls_batch, dim=0).to(_device)

    def _test_crf_model(self, dataloader: DataLoader, ml_set: str):
        """Tests the CRF model with the dev/test sets."""

        correct = 0
        existing = 0

        for inp_lex, inp_emb, inp_ctx, tar_ctg in tqdm(dataloader, desc=f'Eval'):
            with torch.inference_mode():
                scores, batch_labels_list = self._crf_model(
                    x=(inp_lex, inp_emb, inp_ctx))
            # end with
            
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
        logger.info(f'Acc = {acc:.5f} on {ml_set}')

    def _test_cls_model(self, dataloader: DataLoader, ml_set: str):
        """Tests the CLS model with the dev/test sets."""

        correct = 0
        existing = 0

        for inp_lex, inp_emb, inp_ctx, _, tar_cls in tqdm(dataloader, desc=f'Eval'):
            with torch.inference_mode():
                _, outputs_cls = self._cls_model(x=(inp_lex, inp_emb, inp_ctx))
            # end with

            outputs_cls = torch.exp(outputs_cls)
            predicted_labels = torch.argmax(outputs_cls, dim=2)
            correct_labels = \
                (predicted_labels == tar_cls).to(dtype=torch.int32)
            
            correct += torch.sum(correct_labels).item()
            existing += predicted_labels.numel()
        # end for

        acc = correct / existing
        logger.info(f'Acc = {acc:.5f} on {ml_set}')

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
                logger.info(f'Batch {counter}/{len(dataloader)} loss: {average_running_loss:.5f}')
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        logger.info(f'Average epoch {epoch} loss: {average_epoch_loss:.5f}')

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
                logger.info(f'Batch {counter}/{len(dataloader)} loss: {average_running_loss:.5f}')
                epoch_loss.append(average_running_loss)
                running_loss = 0.
            # end if
        # end for i

        average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        logger.info(f'Average epoch {epoch} loss: {average_epoch_loss:.5f}')

    def _build_cls_model(self,
                         emb_input_vector_size: int,
                         lex_input_vector_size: int,
                         ctx_input_vector_size: int,
                         msd_encoding_vector_size: int,
                         output_msd_size: int,
                         drop_prob: float = 0.33) -> CLSModel:
        return CLSModel(emb_input_vector_size,
                        lex_input_vector_size, ctx_input_vector_size,
                        msd_encoding_vector_size, output_msd_size,
                        drop_prob)

    def _build_crf_model(self,
                         emb_input_vector_size: int,
                         lex_input_vector_size: int,
                         ctx_input_vector_size: int,
                         drop_prob: float = 0.33) -> CRFModel:
        return CRFModel(
            emb_input_vector_size,
            self._msd.get_ctag_inventory(),
            lex_input_vector_size, ctx_input_vector_size,
            drop_prob)

    def _build_crf_io_tensor(self,
                             sentence: List[Tuple[str, str, List[str]]]) -> \
                                Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]:
        """Builds the CRF features tensors for one sentence."""

        tok_sentence = [(x[0], self._tokenizer.tag_word(x[0]))
                        for x in sentence]
        bert_sentence_features = \
            self._crf_ro_bert.bert_embeddings(tokens=tok_sentence)

        sentence_lex_features = []
        sentence_embeddings = []
        sentence_context = []
        sentence_ctags = []

        for j in range(len(sentence)):
            parts = sentence[j]
            word = parts[0]
            msd = parts[1]
            ctag = self._msd.msd_to_ctag(msd)

            if not self._msd.is_valid_ctag(ctag):
                logger.warning(f"Unknown CTAG [{ctag}] for MSD [{msd}]")
            # end if

            feats = parts[2]
            tlabel = self._tokenizer.tag_word(word)
            y_out_idx = self._msd.ctag_to_idx(ctag)

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
            x_emb = bert_sentence_features[j]
            # Computing external features for word
            x_ctx = self._rofeatures.get_context_feature_vector(feats)

            sentence_lex_features.append(torch.tensor(
                x_lex_ctag, dtype=torch.float32).view(1, -1))
            sentence_embeddings.append(x_emb.view(1, -1))
            sentence_context.append(torch.tensor(
                x_ctx, dtype=torch.float32).view(1, -1))
            sentence_ctags.append(y_out_idx)
        # end j

        return torch.cat(sentence_lex_features, dim=0).to(_device), \
            torch.cat(sentence_embeddings, dim=0).to(_device), \
            torch.cat(sentence_context, dim=0).to(_device), \
            torch.tensor(sentence_ctags, dtype=torch.long).to(_device)

    def _build_cls_io_tensor(self,
                             sample: List[Tuple[str, str, List[str]]]) -> Tuple[torch.Tensor,
                                                                                torch.Tensor, torch.Tensor,
                                                                                torch.Tensor, torch.Tensor]:
        """Builds the CLS features tensors for one sample."""

        tok_sample = [(x[0], self._tokenizer.tag_word(x[0]))
                      for x in sample]
        bert_sample_features = \
            self._cls_ro_bert.bert_embeddings(tokens=tok_sample)
        xlex_msd_tensors = []
        xemb_tensors = []
        xctx_tensors = []
        y_tensors_enc = []
        y_cls = []

        for j in range(len(sample)):
            parts = sample[j]
            word = parts[0]
            msd = parts[1]
            feats = parts[2]
            tlabel = self._tokenizer.tag_word(word)
            y_in = self._msd.msd_input_vector(msd)
            y_out = self._msd.msd_to_idx(msd)
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

            # Computing BERT embedding for word
            x_emb = bert_sample_features[j]
            # Computing external features for word
            x_ctx = self._rofeatures.get_context_feature_vector(feats)

            xlex_msd_tensors.append(torch.tensor(x_lex_msd, dtype=torch.float32).view(1, -1))
            xemb_tensors.append(x_emb.view(1, -1))
            xctx_tensors.append(torch.tensor(x_ctx, dtype=torch.float32).view(1, -1))
            y_tensors_enc.append(torch.tensor(y_in, dtype=torch.float32).view(1, -1))
            y_cls.append(y_out)
        # end j

        return torch.cat(xlex_msd_tensors, dim=0).to(_device), \
            torch.cat(xemb_tensors, dim=0).to(_device), \
            torch.cat(xctx_tensors, dim=0).to(_device), \
            torch.cat(y_tensors_enc, dim=0).to(_device), \
            torch.tensor(y_cls, dtype=torch.long).to(_device)

    def _build_unicode_props(self, data_samples: List[List[Tuple[str, str, List[str]]]]):
        for sample in tqdm(data_samples, desc='Unicode props'):
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

    def read_tagged_file(self, file: str) -> List[List[Tuple[str, str, List[str]]]]:
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
                    logger.info(f"Line [{line_count}] in file [{file}] is not well-formed!")
                else:
                    current_sentence.append((parts[0], parts[2]))
                # end if
            # end all lines
        # end with

        return sentences

    def _get_predicted_msd(self, y_pred: np.ndarray, epsilon: float = 2e-5) -> List[Tuple[str, float]]:
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

    def _most_prob_msd(self, word: str, y_pred: np.ndarray, tt_msds: List[str]) -> Tuple[str, float, bool, List[str]]:
        # 0. Get fixed tag assignment, if any
        fixed_msd = self._fixed_tag_assignments(word)

        if fixed_msd:
            return fixed_msd, 1., True, []
        # end if

        # 1. Get the model predicted MSDs
        msd_predictions = self._get_predicted_msd(y_pred)
        cls_only_pred_msds = [m for m, _ in msd_predictions]
        crf_cls_agree = False

        # 1.1 Update MSD predictions with tiered tagging MSDs
        for ttm in tt_msds:
            if ttm not in cls_only_pred_msds:
                idx = self._msd.msd_to_idx(ttm)
                msd_p = y_pred[idx]
                msd_predictions.append((ttm, msd_p))
            else:
                crf_cls_agree = True
            # end if
        # end for

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
            # 4. Model predicted MSD is in the extended ambiguity class. Hurray!
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

            return best_pred_msd, best_pred_msd_p, crf_cls_agree, cls_only_pred_msds
        elif known_word_msds:
            computed_word_msds = list(known_word_msds)
        else:
            computed_word_msds = best_pred_msds
        # end if

        # 5. Model predicted MSD is not in the extended ambiguity class.
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

        log_once(f"Word [{word}] -> got suboptimal MSD [{best_acls_msd}/{best_acls_msd_p:.2f}]",
                 calling_fn='RoPOSTagger._most_prob_msd', log_level=logging.DEBUG)

        return best_acls_msd, best_acls_msd_p, crf_cls_agree, cls_only_pred_msds

    def _tiered_tagging(self, word: str, ctag: str) -> List[str]:
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
            log_once(
                f"_tiered_tagging[standard]: word '{word}' got MSDs [{', '.join(result_aclass)}] for CTAG [{ctag}]",
                'RoPOSTagger._tiered_tagging', logging.DEBUG)
            return result_aclass
        # end if

        amend_aclass = self._lexicon.amend_ambiguity_class(word, both_aclass)
        result_aclass = list(amend_aclass.intersection(mclass))

        if result_aclass:
            log_once(
                f"_tiered_tagging[extended]: word '{word}' got MSDs [{', '.join(result_aclass)}] for CTAG [{ctag}]",
                'RoPOSTagger._tiered_tagging', logging.DEBUG)
            return result_aclass
        else:
            # If result_aclass is empty, tiered tagging failed:
            # CTAG is very different than ambiguity classes!
            list_amend_aclass = list(amend_aclass)
            log_once(
                f"_tiered_tagging[error]: word '{word}' got MSDs [{', '.join(list_amend_aclass)}] for CTAG [{ctag}]",
                'RoPOSTagger._tiered_tagging', logging.DEBUG)
            return list_amend_aclass
        # end if

    def tag_sentence(self, sentence: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
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

    def _eval_sentence(self, sentence: List[Tuple[str, str, List[str]]],
                       debug_fh = None) -> List[Tuple[str, str, float]]:
        """Only works on a single sentence. Main method that calls both the
        `self._cls_model` and `self._crf_model` to do the heavy lifting."""

        # 1 Get the input tensor for CLS, for the whole sentence.
        # y_pred_cls is the POS tagger sequence with RNNs, one-hot
        cls_x_lex, cls_x_emb, cls_x_ctx, y_enc, y_cls = \
            self._cls_collate_fn(batch=[sentence])

        with torch.inference_mode():
            out_enc, out_cls = \
                self._cls_model(x=(cls_x_lex, cls_x_emb, cls_x_ctx))
        # end with

        y_pred_cls = torch.exp(out_cls).cpu().detach().numpy()
        viterbi_sequence = []

        if RoPOSTagger._conf_with_tiered_tagging:
            # 2 Get the input tensors for CRF, for the whole sentence.
            # viterbi_sequence is the CTAG index sequence with CRFs
            crf_x_lex, crf_x_emb, crf_x_ctx, y_crf = \
                self._crf_collate_fn(batch=[sentence])

            with torch.inference_mode():
                scores, batch_labels_list = self._crf_model(
                    x=(crf_x_lex, crf_x_emb, crf_x_ctx))
            # end with

            viterbi_sequence = batch_labels_list[0]
            assert len(viterbi_sequence) == len(sentence)
        # end if

        tagged_sentence = []

        for i in range(len(sentence)):
            word = sentence[i][0]
            gold_msd = sentence[i][1]
            y1 = y_pred_cls[0, i, :]
            crf_msds = []

            if RoPOSTagger._conf_with_tiered_tagging:
                y2 = viterbi_sequence[i]
                ctag = self._msd.idx_to_ctag(y2)
                crf_msds = self._tiered_tagging(word, ctag)
            # end if with tiered tagging

            best_msd, msd_p, both_agree, cls_msds = \
                self._most_prob_msd(word, y1, tt_msds=crf_msds)

            if debug_fh:
                if not crf_msds:
                    crf_msds = ['n/a']
                # end if

                if gold_msd == best_msd:
                    print(
                        f"{word}\t{gold_msd}\tp={msd_p:.5f}", file=debug_fh)
                else:
                    if both_agree:
                        print(
                            f"{word}\t{gold_msd}\tBEST:{best_msd}\tp={msd_p:.5f}\t"
                            f"CRF:{','.join(crf_msds)} == CLS:{','.join(cls_msds)}", file=debug_fh)
                    else:
                        print(
                            f"{word}\t{gold_msd}\tBEST:{best_msd}\tp={msd_p:.5f}\t"
                            f"CRF:{','.join(crf_msds)} != CLS:{','.join(cls_msds)}", file=debug_fh)
                    # end if
                # end if
            # end if

            tagged_sentence.append((word, best_msd, msd_p))
        # end for

        if debug_fh:
            print(file=debug_fh, flush=True)
        # end if

        return tagged_sentence

    def test_on_sentence_set(self, sentences: List[List[Tuple[str, str, List[str]]]], ml_type: str):
        all_count = 0
        correct_count = 0
        debug_output_file = os.path.join(TAGGER_MODEL_FOLDER, f'tagger-debug-{ml_type}.txt')

        with open(debug_output_file, mode='w', encoding='utf-8') as f:
            for sentence in tqdm(sentences, desc=f'Eval'):
                tagged_sentence = \
                    tag._eval_sentence(sentence, debug_fh=f)
        
                for i in range(len(sentence)):
                    if tagged_sentence[i][1] == sentence[i][1]:
                        correct_count += 1
                    # end if
                # end for

                all_count += len(sentence)
            # end all sentences
        # end with

        acc = correct_count / all_count
        logger.info(f'On [{ml_type}] set: Acc = {acc:.5f}')
        self._romorphology.save_cache()


if __name__ == '__main__':
    # Use this module to train the POS tagger.
    lex = Lex()
    tok = RoTokenizer(lex)
    spl = RoSentenceSplitter(lex, tok)
    spl.load()
    mor = RoInflect(lex)
    mor.load()
    tag = RoPOSTagger(lex, tok, mor, spl)

    training_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-train.tab")
    logger.info(f"Reading training file [{training_file}]")
    training = tag.read_tagged_file(training_file)

    development_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-dev.tab")
    logger.info(f"Reading development file [{development_file}]")
    development = tag.read_tagged_file(development_file)

    testing_file = os.path.join(
        "data", "training", "tagger", "ro_rrt-ud-test.tab")
    logger.info(f"Reading testing file [{testing_file}]")
    testing = tag.read_tagged_file(testing_file)

    tag.train(train_sentences=training,
              dev_sentences=development, test_sentences=testing)
    tag.test_on_sentence_set(sentences=development, ml_type='dev')
    tag.test_on_sentence_set(sentences=testing, ml_type='test')
