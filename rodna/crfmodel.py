# Bi-LSTM Conditional Random Field
# Author: Robert Guthrie
# Adapted by Radu Ion for RODNA
# Implementation adapted from https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

from random import shuffle
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from utils.MSD import MSD
from .embeddings import RoWordEmbeddings
from . import _device

# The tag in front of the sentence
START_TAG = MSD.get_start_end_tags('start')
# The tag at the end of the sentence
STOP_TAG = MSD.get_start_end_tags('end')


def argmax(vec: Tensor) -> int:
    """Return the argmax as a Python int."""

    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec: Tensor):
    """Compute log sum exp in a numerically stable way for the forward algorithm."""

    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast))).to(device=_device)


class CRFModelDataset(Dataset):
    """This is a dataset for the BiGRUCRF model below."""

    def __init__(self, lex_feats: list,
            emb_feats: list, ctx_feats: list, y_ctags: list):
        super().__init__()

        # len(emb_feats) is the number of sentences in this data set
        assert len(lex_feats) == len(emb_feats)
        assert len(emb_feats) == len(ctx_feats)
        assert len(ctx_feats) == len(y_ctags)
        
        # Group tensors by sentence length, for batch processing.
        the_data = {}
        sentence_lengths = []

        for i in range(len(lex_feats)):
            i_len = lex_feats[i].shape[0]

            if i_len not in the_data:
                the_data[i_len] = {'lex': [],
                                     'emb': [], 'ctx': [], 'yct': []}
                sentence_lengths.append(i_len)
            # end if

            the_data[i_len]['lex'].append(lex_feats[i])
            the_data[i_len]['emb'].append(emb_feats[i])
            the_data[i_len]['ctx'].append(ctx_feats[i])
            the_data[i_len]['yct'].append(y_ctags[i])
        # end for

        # Get a random order of sentence lenghts order
        shuffle(sentence_lengths)
        self._data = []

        for i_len in sentence_lengths:
            for i in range(len(the_data[i_len]['lex'])):
                self._data.append((
                    the_data[i_len]['lex'][i],
                    the_data[i_len]['emb'][i],
                    the_data[i_len]['ctx'][i],
                    the_data[i_len]['yct'][i]
                ))
            # end for
        # end for

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class CRFModel(nn.Module):
    """This implements a POS tagger based on a BiGRU network
    supplying the emmision features and a CRF layer to choose
    the optimal tag."""

    _conf_rnn_size = 256

    def __init__(self,
                 embeds: RoWordEmbeddings, tag_to_ix: dict,
                 lex_input_vector_size: int, ctx_input_vector_size: int,
                 runtime: bool,
                 drop_prob: float = 0.33):
        super().__init__()
        # START_TAG and STOP_TAG are assumed to be in tag_to_ix
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding.from_pretrained(
            embeds.get_embeddings_weights(runtime),
            freeze=False)
        self.gru = nn.GRU(
            embeds.get_vector_length() + lex_input_vector_size,
            CRFModel._conf_rnn_size,
            batch_first=True,
            bidirectional=True)
        self.drop = nn.Dropout(p=drop_prob)

        # Maps the output of the GRU into tag space.
        self.hidden2tag = nn.Linear(
            2 * CRFModel._conf_rnn_size + ctx_input_vector_size, self.tagset_size)

        # Matrix of transition parameters. Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.to(device=_device)

    def _init_hidden(self, bsize: int):
        return torch.randn(2, bsize, CRFModel._conf_rnn_size).to(device=_device)

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device=_device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # end for
            forward_var = torch.cat(alphas_t).view(1, -1)
        # end for

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _get_gru_features(self, x):
        """This is the batched version."""

        x_lex, x_emb, x_ctx = x
        bs = x_lex.shape[0]
        self.hidden = self._init_hidden(bsize=bs)

        out = self.word_embeds(x_emb)
        out = torch.cat([out, x_lex], dim=2)
        out, self.hidden = self.gru(out, self.hidden)
        out = self.drop(out)
        out = torch.cat([out, x_ctx], dim=2)
        feats = self.hidden2tag(out)

        return feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device=_device)
        tags = torch.cat([
            torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device=_device),
            tags]).to(device=_device)

        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # end for

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the Viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device=_device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # end for

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        # end for

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # end for

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()

        return path_score, best_path

    def neg_log_likelihood(self, x, batch_tags):
        """`x` are the input features and `tags` are the gold
        standard tags for the sentence."""

        batch_feats = self._get_gru_features(x)
        batch_forward_score = torch.zeros(
            1, dtype=torch.float32).to(device=_device)
        batch_gold_score = torch.zeros(
            1, dtype=torch.float32).to(device=_device)

        for i in range(batch_feats.shape[0]):
            feats = batch_feats[i, :, :]
            tags = batch_tags[i, :]
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            batch_forward_score = batch_forward_score + forward_score
            batch_gold_score = batch_gold_score + gold_score
        # end for

        return batch_forward_score - batch_gold_score

    def forward(self, x):
        # 1. Get the emission scores from the BiGRU
        batch_feats = self._get_gru_features(x)

        # 2. Find the best path, given the features.
        scores = []
        tag_sequences = []
        
        for i in range(batch_feats.shape[0]):
            feats = batch_feats[i, :, :]
            score, tag_seq = self._viterbi_decode(feats)
            scores.append(score)
            tag_sequences.append(tag_seq)
        # end for

        return scores, tag_sequences
