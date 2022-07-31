import sys
import os
import torch
from torch import Tensor
from inspect import stack
from config import EMBEDDING_VOCABULARY_FILE
from utils.Lex import Lex
from . import _device

zero_word = '_ZERO_'
unk_word = '_UNK_'
start_word = '_START_'
end_word = '_END_'


class RoWordEmbeddings(object):
    """An external word embeddings representation, relying
    on the embeddings loaded by the lexicon object."""

    def __init__(self, lexicon: Lex) -> None:
        self._lexicon = lexicon
        self._wembdim = self._lexicon.get_word_embeddings_size()
        self._wembvoc = {}

    def load_ids(self):
        """Call this at runtime, before anything else."""

        if os.path.exists(EMBEDDING_VOCABULARY_FILE):
            with open(EMBEDDING_VOCABULARY_FILE, mode='r', encoding='utf-8') as f:
                first_line = True

                for line in f:
                    line = line.rstrip()

                    if first_line:
                        assert int(line) == self._wembdim
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
        self._wembmat = []

        def _add_word(word: str, vec: list):
            if word not in self._wembvoc:
                self._wembmat.append(vec)
                self._wembvoc[word] = len(self._wembmat) - 1
            else:
                print(stack()[0][3] + ": word '{0}' is duplicated!".format(
                    word), file=sys.stderr, flush=True)
            # end if
        # end def

        # The padding word, also the unknown word.
        _add_word(zero_word, [0] * self._wembdim)
        # The vector for the unknown word
        _add_word(unk_word, [0.5] * self._wembdim)
        # The vector for the start sentence anchor
        _add_word(start_word, [1.0] * self._wembdim)
        # The vector for the end sentence anchor
        _add_word(end_word, [1.0] * self._wembdim)

        for word in sorted(word_list):
            wwe = self._lexicon.get_word_embeddings_exact(word)

            if wwe:
                _add_word(word, wwe)
            else:
                lc_word = word.lower()
                wwe = self._lexicon.get_word_embeddings_exact(lc_word)

                if wwe:
                    _add_word(lc_word, wwe)
                else:
                    _add_word(word, [0.5] * self._wembdim)
                # end if
            # end if
        # end for

        assert len(self._wembvoc) == len(self._wembmat)
        self._wembvsz = len(self._wembvoc)

        # Add extra words to this vocabulary and save them
        with open(EMBEDDING_VOCABULARY_FILE, mode='w', encoding='utf-8') as f:
            print("{0!s}".format(self._wembdim), file=f)

            for word in sorted(self._wembvoc.keys()):
                print("{0}\t{1!s}".format(word, self._wembvoc[word]), file=f)
            # end for
        # end with

    def get_embeddings_weights(self, runtime: bool) -> Tensor:
        if runtime:
            # At runtime, load the trained weights
            return torch.zeros(self._wembvsz, self._wembdim, dtype=torch.float32).to(device=_device)
        else:
            return torch.tensor(self._wembmat, dtype=torch.float32).to(device=_device)
        # end if

    def get_word_id(self, word: str) -> int:
        if word in self._wembvoc:
            return self._wembvoc[word]
        elif word.lower() in self._wembvoc:
            return self._wembvoc[word.lower()]
        else:
            return self._wembvoc[unk_word]
        # end if

    def get_vector_length(self) -> int:
        return self._wembdim

    def get_vocabulary_size(self) -> int:
        return self._wembvsz
