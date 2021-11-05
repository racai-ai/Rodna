import sys
import re
import gzip
from inspect import stack
from typing import Pattern
import numpy as np
from .MSD import MSD
from config import TBL_WORDFORM_FILE, EXTERNAL_WORD_EMBEDDINGS_FILE
from utils.errors import print_error


class Lex(object):
    """This class will read in the lexicon (in tbl.wordform format)"""

    # Pronouns, Determiners, Particles, Adpositions, Conjunctions, Abbreviations, Numerals
    # proper nouns, numerals, adverbs, not general.
    _mwe_pos_pattern = re.compile("^([PDQSCYMI]|Np|R[^g])")
    _abbr_pos_pattern = re.compile("^Y")
    content_word_pos_pattern = re.compile("^([YNAM]|Vm|Rg)")
    _comm_pattern = re.compile("^\\s*#")
    # Length of the affixes to do affix analysis.
    _prefix_length = 5
    _suffix_length = 5
    sentence_case_pattern = re.compile("^[A-ZȘȚĂÎÂ][a-zșțăîâ_-]+$")
    mixed_case_pattern = re.compile(
        "^[a-zA-ZșțăîâȘȚĂÎÂ]*[a-zșțăîâ-][A-ZȘȚĂÎÂ][a-zA-ZșțăîâȘȚĂÎÂ-]*$")
    upper_case_pattern = re.compile("^[A-ZȘȚĂÎÂ_-]+$")
    number_pattern = re.compile("^([0-9]+|[0-9]+[.,][0-9]+)$")
    bullet_number_pattern = re.compile("^(.*[0-9][./-].+|.*[./-][0-9].*)$")
    _case_patterns = [
        # Lower
        re.compile("^[a-zșțăîâ_-]+$"),
        # Sentence
        sentence_case_pattern,
        # Upper
        upper_case_pattern,
        # Mixed
        mixed_case_pattern,
        # Code
        re.compile("^[A-Z][A-ZȘȚĂÎÂ-]*[0-9][A-Za-z0-9șțăîâȘȚĂÎÂ-]*$"),
        # Punctuation
        re.compile("^\\W+$"),
    ]

    def __init__(self):
        # Dictionary lexicon
        self._lexicon = {}
        # The MSD representation
        self._msd = MSD()
        # Word embeddings lexicon
        self._wdembed = {}
        # The size of the word embedding vector
        self._wembdim = 0
        # Possible POSes for each word in the lexicon
        self._possibletags = {"UNK": 0}
        self._tagid = 1
        # Maximum length of a multi-word expression (MWE)
        self._maxmwelen = 2
        self._maxabbrlen = 2
        self._mwefirstword = set()
        self._abbrfirstword = set()
        # The set of 'a fi' word forms, lower-cased
        self._tobewordforms = set()
        self._canwordforms = set()
        # For abbreviations with 2 tokens, e.g. 'etc.', 'nr.', etc.
        # They have effectively one token
        self._abbrfirstword1 = set()
        self.longestwordlen = 20
        self._prefixes = {}
        self._suffixes = {}
        self._read_word_embeddings()
        self._read_tbl_wordform()
        self._remove_abbr_first_words_that_are_lex_words()
        self._add_no_diac_words_to_embeddings()

    def _read_word_embeddings(self):
        if EXTERNAL_WORD_EMBEDDINGS_FILE.endswith(".gz"):
            f = gzip.open(EXTERNAL_WORD_EMBEDDINGS_FILE,
                          mode="rt", encoding="utf-8")
        else:
            f = open(EXTERNAL_WORD_EMBEDDINGS_FILE, mode="r", encoding="utf-8")
        # end if

        line = f.readline()
        parts = line.strip().split()
        # Add the unknown word to the size
        self._wembdim = int(parts[1])
        counter = 0

        for line in f:
            counter += 1

            if counter % 100000 == 0:
                print(stack()[0][3] + ": read {0!s} lines from file {1}".format(counter, EXTERNAL_WORD_EMBEDDINGS_FILE),
                      file=sys.stderr, flush=True)

            parts = line.strip().split()
            word = parts.pop(0)

            if self._wembdim != len(parts):
                print(stack()[0][3] + ": incorrect dimension of {0} vs. {1} in file {2} at line {3}".format(len(parts),
                                                                                                            self._wembdim, EXTERNAL_WORD_EMBEDDINGS_FILE, counter), file=sys.stderr, flush=True)
                continue
            # end if

            self._wdembed[word] = [float(x) for x in parts]
        # end for

        f.close()

    def get_msd_object(self) -> MSD:
        return self._msd

    def _remove_abbr_first_words_that_are_lex_words(self) -> None:
        """We don't want to tag 'loc.' in e.g. 'au adus-o pe loc.' as an abbreviation."""

        for word in self._abbrfirstword1:
            if not self.is_lex_word(word) or \
                self.is_msd_rxonly_word(word, Lex._abbr_pos_pattern):
                self._abbrfirstword.add(word)
            # end if
        # end for

    def _add_no_diac_words_to_embeddings(self) -> None:
        nd_embed = {}

        for word in self._wdembed:
            nd_word = self._get_romanian_word_with_no_diacs(word)

            if word != nd_word and nd_word not in self._wdembed:
                nd_embed[nd_word] = self._wdembed[word]
            # end if
        # end for

        for word in nd_embed:
            self._wdembed[word] = nd_embed[word]
        # end for

    @staticmethod
    def repl_sgml_wih_utf8(word: str) -> str:
        word = word.replace("&abreve;", "ă")
        word = word.replace("&acirc;", "â")
        word = word.replace("&icirc;", "î")
        word = word.replace("&scedil;", "ș")
        word = word.replace("&tcedil;", "ț")
        word = word.replace("&Abreve;", "Ă")
        word = word.replace("&Acirc;", "Â")
        word = word.replace("&Icirc;", "Î")
        word = word.replace("&Scedil;", "Ș")
        word = word.replace("&Tcedil;", "Ț")

        return word

    def _read_tbl_wordform(self) -> None:
        counter = 0
        word_lengths = {}

        with open(TBL_WORDFORM_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                counter += 1

                if counter % 100000 == 0:
                    print(stack()[0][3] + ": read {0!s} lines from file {1!s}".format(
                        counter, TBL_WORDFORM_FILE), file=sys.stderr, flush=True)

                if Lex._comm_pattern.search(line) != None:
                    continue
                # end if

                parts = line.split()

                if len(parts) == 3:
                    word = parts[0]
                    word = Lex.repl_sgml_wih_utf8(word)
                    wl = len(word)

                    # Word length analysis
                    if wl not in word_lengths:
                        word_lengths[wl] = 1
                    else:
                        word_lengths[wl] += 1
                    # end if

                    lemma = parts[1]

                    if lemma == '=':
                        lemma = word
                    else:
                        lemma = Lex.repl_sgml_wih_utf8(lemma)
                    # end if

                    # 'îi' is also a pronoun, do not count it a to be form
                    # It's a rarely used form for 'este'
                    if lemma.lower() == 'fi' and word.lower() != "îi":
                        self._tobewordforms.add(word.lower())
                    # end if

                    if lemma.lower() == 'putea':
                        self._canwordforms.add(word.lower())
                    # end if

                    msd = parts[2]
                    pos = MSD.get_msd_pos(msd)

                    if pos not in self._possibletags:
                        self._possibletags[pos] = self._tagid
                        self._tagid += 1
                    # end if

                    diac_word = word
                    no_diac_word = self._get_romanian_word_with_no_diacs(word)

                    self._add_word_to_lex(diac_word, msd)
                    self._add_word_to_lex(no_diac_word, msd)

                    # It's an abbreviation
                    if word.endswith(".") and Lex._abbr_pos_pattern.match(msd):
                        parts = word.split('.')
                        # Delete the last empty token.
                        parts.pop()

                        if len(parts) > self._maxabbrlen:
                            self._maxabbrlen = len(parts)
                        # end if

                        if len(parts) == 1:
                            self._abbrfirstword1.add(parts[0])
                        else:
                            self._abbrfirstword.add(parts[0])
                        # end if
                    # end if

                    # It's a MWE
                    if '_' in word and Lex._mwe_pos_pattern.match(msd):
                        parts = word.split('_')

                        if len(parts) > self._maxmwelen:
                            self._maxmwelen = len(parts)
                        # end if

                        self._mwefirstword.add(parts[0])
                    # end if

                    if Lex.content_word_pos_pattern.match(msd):
                        # Affix analysis
                        if len(word) >= Lex._prefix_length:
                            prefix = word[0:Lex._prefix_length]

                            for i in range(len(prefix)):
                                i_prefix = prefix[0: i + 1]

                                if i_prefix not in self._prefixes:
                                    self._prefixes[i_prefix] = {}
                                # end if

                                if msd not in self._prefixes[i_prefix]:
                                    self._prefixes[i_prefix][msd] = 1
                                else:
                                    self._prefixes[i_prefix][msd] += 1
                                # end if
                            # end for
                        # end if

                        if len(word) >= Lex._suffix_length:
                            suffix = word[-Lex._suffix_length:]

                            for i in range(1, len(suffix) + 1):
                                i_suffix = suffix[-i:]

                                if i_suffix not in self._suffixes:
                                    self._suffixes[i_suffix] = {}
                                # end if

                                if msd not in self._suffixes[i_suffix]:
                                    self._suffixes[i_suffix][msd] = 1
                                else:
                                    self._suffixes[i_suffix][msd] += 1
                                # end if
                            # end for
                        # end if
                    # end if content word for affix analysis
                # end if parts has 3 elements
            # end for all lines
        # end with

        max_wl = 0

        for wl in word_lengths:
            if wl > max_wl and word_lengths[wl] > 1000:
                max_wl = wl
            # end if
        # end for

        self.longestwordlen = max_wl

    def is_to_be_word(self, word: str) -> bool:
        return word.lower() in self._tobewordforms

    def is_can_word(self, word: str) -> bool:
        return word.lower() in self._canwordforms

    def has_ambiguity_class(self, word: str, msd1: str, msd2: str) -> bool:
        word_msds = self.get_word_ambiguity_class(word)
        is_msd1 = False
        is_msd2 = False

        for msd in word_msds:
            if msd == msd1 or msd.startswith(msd1):
                is_msd1 = True
            # end if

            if msd == msd2 or msd.startswith(msd2):
                is_msd2 = True
            # end if
        # end if

        return is_msd1 and is_msd2

    def can_be_msd(self, word: str, msd: str) -> bool:
        word_msds = self.get_word_ambiguity_class(word)

        for m in word_msds:
            if m == msd or m.startswith(msd):
                return True
            # end if
        # end for

        return False

    @staticmethod
    def _get_romanian_word_with_no_diacs(word: str) -> str:
        word = word.replace("ă", "a")
        word = word.replace("î", "i")
        word = word.replace("â", "a")
        word = word.replace("ș", "s")
        word = word.replace("ş", "s")
        word = word.replace("ț", "t")
        word = word.replace("ţ", "t")

        word = word.replace("Ă", "A")
        word = word.replace("Î", "I")
        word = word.replace("Â", "A")
        word = word.replace("Ș", "S")
        word = word.replace("Ş", "S")
        word = word.replace("Ț", "T")
        word = word.replace("Ţ", "T")

        return word

    def _add_word_to_lex(self, word: str, msd: str) -> None:
        if word not in self._lexicon:
            self._lexicon[word] = set()
        # end if

        if msd not in self._lexicon[word]:
            self._lexicon[word].add(msd)
        # end if

    def is_msd_rxonly_word(self, word: str, pattern: Pattern) -> bool:
        """Tests if the given word has MSDs which ALL
        match the given MSD regular expression."""

        all_msds = set()

        if word in self._lexicon:
            all_msds.update(self._lexicon[word])
        # end if

        if word.lower() in self._lexicon:
            all_msds.update(self._lexicon[word.lower()])
        # end if

        if not all_msds:
            return False
        # end if

        for msd in all_msds:
            if not pattern.match(msd):
                return False
            # end if
        # end for

        return True

    def is_lex_word(self, word: str, exact_match: bool = False) -> bool:
        """Tests if word is in this lexicon or not."""

        if exact_match:
            return word in self._lexicon
        else:    
            return word in self._lexicon or word.lower() in self._lexicon

    def is_wemb_word(self, word: str) -> bool:
        """Tests if word is in the word embeddings or not."""

        return word in self._wdembed or word.lower() in self._wdembed

    def is_mwe_first_word(self, word: str) -> bool:
        """Tests if word can start a multi-word expression."""

        return word in self._mwefirstword or word.lower() in self._mwefirstword

    def is_abbr_first_word(self, word: str) -> bool:
        """Tests if word can start an abbreviation."""

        return word in self._abbrfirstword or word.lower() in self._abbrfirstword

    def accept_phrasal_token(self, word: str, label: str) -> bool:
        phr_ok = False
        pattern = None

        if label == "MWE":
            pattern = Lex._mwe_pos_pattern
        elif label == "ABBR":
            pattern = Lex._abbr_pos_pattern
        # end if

        if word in self._lexicon:
            for msd in self._lexicon[word]:
                if pattern.match(msd):
                    phr_ok = True
                    break
                # end if
            # end for msd
        # end if

        if phr_ok:
            return phr_ok
        # end if

        word = word.lower()

        if word in self._lexicon:
            for msd in self._lexicon[word]:
                if pattern.match(msd):
                    phr_ok = True
                    break
                # end if
            # end for msd
        # end if

        return phr_ok

    def get_word_ambiguity_class(self, word: str, exact_match: bool = False) -> list:
        all_msds = set()

        if word in self._lexicon:
            all_msds.update(self._lexicon[word])
        # end if

        if exact_match:
            return list(all_msds)
        # end if

        if word.lower() in self._lexicon:
            all_msds.update(self._lexicon[word.lower()])
        # end if

        return list(all_msds)

    def get_unknown_ambiguity_class(self, word: str) -> list:
        """Used as a backup of RoInflect neural network."""

        prefix_msds = self._get_possible_msds_from_prefix(word)
        suffix_msds = self._get_possible_msds_from_suffix(word)
        affix_msds = prefix_msds.union(suffix_msds)

        # See if we can add a Np-type MSD to the ambiguity class.
        if Lex.sentence_case_pattern.match(word) or \
                Lex.mixed_case_pattern.match(word):
            np_msd_present = False

            # 1. Np is already present.
            for msd in affix_msds:
                if msd.startswith('Np'):
                    np_msd_present = True
                # end if
            # end for

            if not np_msd_present:
                affix_msds.add('Np')
                np_msd_added = set()

                # 2. Try and add a more specialized Np MSD.
                for msd in affix_msds:
                    if msd.startswith('Nc') and len(msd) > 2:
                        np_msd = 'Np' + msd[2:]

                        if self._msd.is_valid_msd(np_msd):
                            np_msd_added.add(np_msd)
                        # end if
                    # end if

                    if msd.startswith('Afp') and len(msd) > 3:
                        np_msd = 'Np' + msd[3:]

                        if self._msd.is_valid_msd(np_msd):
                            np_msd_added.add(np_msd)
                        # end if
                    # end if
                # end for

                if np_msd_added:
                    affix_msds = affix_msds.union(np_msd_added)
                # end if
            # end if
        # end if

        word_amb_class = list(affix_msds)

        print_error("unknown word '{0}' has ambiguity class [{1}]".format(
            word, ', '.join([x for x in word_amb_class])), stack()[0][3])

        return word_amb_class

    def _get_possible_msds_from_prefix(self, word: str, min_pref_len: int = 2) -> set:
        """This method minimizes the entropy of possible MSDs for a given prefix."""

        possible_msds = set()
        min_entropy = 1000000.0

        if len(word) >= Lex._prefix_length:
            prefix = word[0:Lex._prefix_length]

            for i in range(len(prefix), min_pref_len, -1):
                i_prefix = prefix[0: i]

                if i_prefix in self._prefixes:
                    msd_probs = np.zeros(
                        len(self._prefixes[i_prefix]), dtype=np.float32)
                    j = 0
                    i_possible_msds = set()
                   
                    for msd in self._prefixes[i_prefix]:
                        msd_probs[j] = self._prefixes[i_prefix][msd]
                        j += 1
                        i_possible_msds.add(msd)
                    # end for

                    msd_probs /= np.sum(msd_probs)
                    msd_entropy = -np.sum(msd_probs * np.log(msd_probs))

                    if msd_entropy < min_entropy:
                        min_entropy = msd_entropy
                        possible_msds = i_possible_msds
                    # end if
                # end if
            # end for
        # end if

        return possible_msds

    def _get_possible_msds_from_suffix(self, word: str, min_suff_len: int = 2) -> set:
        """This method minimizes the entropy of possible MSDs for a given suffix."""

        possible_msds = set()
        min_entropy = 1000000.0

        if len(word) >= Lex._suffix_length:
            suffix = word[-Lex._suffix_length:]

            for i in range(len(suffix), min_suff_len, -1):
                i_suffix = suffix[-i:]

                if i_suffix in self._suffixes:
                    msd_probs = np.zeros(
                        len(self._suffixes[i_suffix]), dtype=np.float32)
                    j = 0
                    i_possible_msds = set()

                    for msd in self._suffixes[i_suffix]:
                        msd_probs[j] = self._suffixes[i_suffix][msd]
                        j += 1
                        i_possible_msds.add(msd)
                    # end for

                    msd_probs /= np.sum(msd_probs)
                    msd_entropy = -np.sum(msd_probs * np.log(msd_probs))

                    if msd_entropy < min_entropy:
                        min_entropy = msd_entropy
                        possible_msds = i_possible_msds
                    # end if
                # end if
            # end for
        # end if

        return possible_msds

    def get_word_embeddings_size(self) -> int:
        return self._wembdim

    def get_word_embeddings_exact(self, word: str) -> list:
        if word in self._wdembed:
            return self._wdembed[word]
        else:
            return []

    def get_word_features(self, word: str) -> np.ndarray:
        """Will get an np.array of lexical features for word."""

        # 1. If word is in lexicon or not and if it is as is
        # or lower-cased.
        features1 = np.zeros(2, dtype=np.float32)

        if word in self._lexicon:
            features1[0] = 1.0
        # end if

        if word.lower() in self._lexicon:
            features1[1] = 1.0
        # end if

        # 1.1 Casing features
        features11 = np.zeros(len(Lex._case_patterns), dtype=np.float32)

        for i in range(len(Lex._case_patterns)):
            patt = Lex._case_patterns[i]

            if patt.match(word):
                features11[i] = 1.0
            # end if
        # end for

        # 2. MSD features for word: the vector of possible tags
        features2 = np.zeros(len(self._possibletags), dtype=np.float32)

        if word not in self._lexicon and word.lower() not in self._lexicon:
            features2[0] = 1.0
        elif word in self._lexicon:
            for msd in self._lexicon[word]:
                pos = MSD.get_msd_pos(msd)
                idx = self._possibletags[pos]
                features2[idx] = 1.0
            # end for
        else:
            for msd in self._lexicon[word.lower()]:
                pos = MSD.get_msd_pos(msd)
                idx = self._possibletags[pos]
                features2[idx] = 1.0
            # end for
        # end if

        # 3. The embedding vector for word
        features3 = np.zeros(self._wembdim, dtype=np.float32)

        if word in self._wdembed:
            features3 = np.array(self._wdembed[word], dtype=np.float32)
        elif word.lower() in self._wdembed:
            features3 = np.array(self._wdembed[word.lower()], dtype=np.float32)
        # end if

        # 4. Concatenate 1, 1.1, 2 and 3
        return np.concatenate((features1, features11, features2, features3))
