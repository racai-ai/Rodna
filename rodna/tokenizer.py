import sys
import re
from inspect import stack
import numpy as np
import unicodedata as uc
from utils.Lex import Lex
from utils.datafile import txt_file_to_string


class RoTokenizer(object):
    """This class will tokenize Romanian texts from input strings.
    The tokenization algorithm is deterministic and rule-based.
    It only works on Romanian."""

    _romanian_word_chars = set(
        "aăâbcdefghiîjklmnopqrsșştțţuvwxyz" +
        "aăâbcdefghiîjklmnopqrsșştțţuvwxyz".upper() +
        "0123456789" +
        "-_")
    _romanian_numbers = set("0123456789")
    _romanian_diacs = set("ăîâșşțţ" + "ăîâșşțţ".upper())
    _romanian_punct_chars = set(",.?!\"’´`‘':;()[]…„”“«»/-_•●·°÷")
    _romanian_sym_chars = set("<>~@#%^&*+={}$\\|§©")
    # \t is reserved. Replace it with ' '
    _romanian_spc_chars = set(" ")
    _romanian_eol_chars = set("\r\n")
    # When splitting the dashed words, e.g. m-ai, prefer these tokens.
    _romanian_keep_dash_words = set([
        "am", "ai", "a", "ați", "au", "al", "ai", "ale",
        "-n", "n-", "o", "un", "-l", "l-",
        "-i", "i-", "e", "-lea", "-ul", "-urile",
        "-ului", "-urilor", "-s"])
    _romanian_reject_mwes = set(["de_a"])
    _romanian_reject_abbrs = set()
    # Plus "JUNK" if nothing else matches.
    # SPACE is used in MWE recognition!
    # EOL comes before SPACE (SPACE contains EOL)!
    _token_classes = [
        "ABBR", "NUM", "RWORD", "MWE",
        "FWORD", "WORD", "EOL", "SPACE",
        "PUNCT", "SYM"]
    _roman_numerals = set([
        "I", "II", "III", "IV", "V",
        "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV",
        "XVI", "XVII", "XVIII", "XIX", "XX",
        "XXI", "XXII", "XXIII", "XXIV", "XXV",
        "XXVI", "XXVII", "XXVIII", "XXIX", "XXX"])
    _number_pattern = re.compile("^[0-9]+$")
    _newline_pattern = re.compile("\\s+$")
    _tab_pattern = re.compile("\\t+")
    _special_pattern = re.compile("^[_-]+$")
    _unicode_char_sets = [
        # Word chars: letters, numbers, - and _
        set(["Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Nl", "No"]).union(
            _romanian_word_chars).union(_romanian_numbers).union(_romanian_diacs),
        # Punctuation chars
        set(["Pf", "Pi", "Pe", "Ps", "Pd", "Pc"]).union(_romanian_punct_chars),
        # Symbol chars
        set(["Sm", "Sc", "Sk"]).union(_romanian_sym_chars),
        # EOL chars
        _romanian_eol_chars,
        # Space chars
        set(["Zs", "Zl", "Zp"]).union(_romanian_spc_chars),
        set()
    ]

    def __init__(self, lexicon: Lex):
        """Takes the Romanian lexicon file and the Romanian word embeddings file."""

        self._lexicon = lexicon

    @staticmethod
    def get_label_features(label: str) -> np.ndarray:
        features = np.zeros(len(RoTokenizer._token_classes), dtype=np.float32)

        for i in range(len(RoTokenizer._token_classes)):
            if label == RoTokenizer._token_classes[i]:
                features[i] = 1.0
                break
            # end if
        # end for

        return features

    def is_rword(self, word: str) -> bool:
        """If a word contains a Romanian diacritic or it is present
        in the Romanian lexicon, it is a Romanian word."""

        if self._lexicon.is_lex_word(word) or \
                self._lexicon.is_lex_word(word.lower()):
            return True
        # end if

        for c in word:
            if c in RoTokenizer._romanian_diacs:
                return True
            # end if
        # end for

        return False

    def is_fword(self, word: str) -> bool:
        """If a word contains a foreign letter, it is a foreign word."""

        for c in word:
            c_cat = uc.category(c)

            if c_cat.startswith("L") and \
                    c not in RoTokenizer._romanian_word_chars:
                return self.is_word(word)
            # end if
        # end for

        return False

    def is_punct(self, word: str) -> bool:
        """If a word contains only Romanian punctuation."""

        for c in word:
            c_cat = uc.category(c)

            if c not in RoTokenizer._romanian_punct_chars and \
               (not c_cat.startswith("P") or c_cat == "Po"):
                return False
            # end if
        # end for

        return True

    def is_sym(self, word: str) -> bool:
        """If a word contains only Romanian symbols."""

        for c in word:
            c_cat = uc.category(c)

            if c not in RoTokenizer._romanian_sym_chars and \
                    (not c_cat.startswith("S") or c_cat == "So"):
                return False
            # end if
        # end for

        return True

    def is_eol(self, word: str) -> bool:
        """If a word contains a single end-of-line character"""
        eol_count = 0

        for c in word:
            if c in RoTokenizer._romanian_eol_chars:
                eol_count += 1
            # end if
        # end for

        if eol_count > 0:
            return True
        # end if

        return False

    def is_space(self, word: str) -> bool:
        """If a word contains only Romanian spaces."""

        for c in word:
            c_cat = uc.category(c)

            if c not in RoTokenizer._romanian_spc_chars and \
                    not c_cat.startswith("Z"):
                return False
            # end if
        # end for

        return True

    def is_num(self, word: str) -> bool:
        """If a word is a natural number or a Roman numeral."""
        if word in RoTokenizer._roman_numerals or \
                word.upper() in RoTokenizer._roman_numerals:
            return True
        # end if

        for c in word:
            c_cat = uc.category(c)

            if c not in RoTokenizer._romanian_numbers and \
                    not c_cat.startswith("N"):
                return False
            # end if
        # end for

        return True

    def is_word(self, word: str) -> bool:
        """If a word contains letters and/or numbers."""

        for c in word:
            c_cat = uc.category(c)

            if not c_cat.startswith("L") and \
                not c_cat.startswith("M") and \
                not c_cat.startswith("N") and \
                c not in RoTokenizer._romanian_word_chars:
                return False
            # end if
        # end for

        if RoTokenizer._special_pattern.search(word) != None:
            return False
        # end if

        return True

    def is_mwe(self, word: str) -> bool:
        """No single word can be a multi-word expression.
        These labels are assigned after the tokenization runs."""

        # Return false, but using word so that pylint shuts up.
        return type(word) is int

    def is_abbr(self, word: str) -> bool:
        """A single word can be an abbreviation in the lexicon."""

        return self._lexicon.is_msd_rxonly_word(word, Lex._abbr_pos_pattern)

    def tag_word(self, word: str) -> str:
        for c in RoTokenizer._token_classes:
            method_name = "is_" + c.lower()
            # Cool Python 3! This one gets the pointer to the method
            # called method_name
            method = getattr(self, method_name)

            if method(word):
                return c
            # end if
        # end for

        return "JUNK"

    def word_is_number(self, word: str) -> bool:
        if RoTokenizer._number_pattern.search(word) != None:
            return True
        # end if

        if word in RoTokenizer._roman_numerals or \
                word.upper() in RoTokenizer._roman_numerals:
            return True
        # end if

        return False

    def word_is_spec_caps(self, word: str) -> bool:
        """Checks if word is of type ABCD or AbCd. Return True
        if it is, False if it's not."""

        pc = None
        allup = True
        mixed = False

        for i in range(len(word)):
            c = word[i]

            if not uc.category(c).startswith('L'):
                return False
            # end if

            if uc.category(c) != 'Lu':
                allup = False
            elif pc is not None and uc.category(pc) == 'Ll':
                mixed = True
            # end if

            pc = c
        # end for

        return allup or mixed

    @staticmethod
    def is_word_label(label: str) -> bool:
        return label == "RWORD" or label == "MWE" or \
            label == "ABBR" or label == "FWORD" or \
            label == "NUM" or label == "WORD"

    @staticmethod
    def is_junk_label(label: str) -> bool:
        return label == "JUNK"

    @staticmethod
    def is_punct_or_sym_label(label: str) -> bool:
        return label == "PUNCT" or label == "SYM"

    @staticmethod
    def is_whitespace_label(label: str) -> bool:
        return label == "EOL" or label == "SPACE"

    def _score_dash_word(self, word: str) -> int:
        score = 0

        if word.lower() in RoTokenizer._romanian_keep_dash_words:
            score += 1
        # end if

        if self._lexicon.is_lex_word(word) or \
            self.word_is_number(word) or \
                self.word_is_spec_caps(word):
            score += 2
        # end if

        return score

    def _decide_dash_split(self, word: str) -> list:
        tokens_dash = []
        word_parts = word.split('-')

        if len(word_parts) == 3:
            left_word = word_parts[0]
            right_word_1 = '-' + word_parts[1]
            right_word_2 = '-' + word_parts[2]

            if self._lexicon.is_lex_word(left_word) and \
                    self._lexicon.is_lex_word(right_word_1) and \
                    self._lexicon.is_lex_word(right_word_2):
                tokens_dash.append((left_word, "RWORD"))
                tokens_dash.append((right_word_1, "RWORD"))
                tokens_dash.append((right_word_2, "RWORD"))

                return tokens_dash
            # end if
        # end if

        if len(word_parts) == 2:
            lw1 = word_parts[0] + '-'
            rw1 = word_parts[1]
            sc1 = self._score_dash_word(lw1) + self._score_dash_word(rw1)
            lw2 = word_parts[0]
            rw2 = '-' + word_parts[1]
            sc2 = self._score_dash_word(lw2) + self._score_dash_word(rw2)
            dash_pair = None
            bsc = 0

            if sc1 >= sc2:
                dash_pair = (lw1, rw1)
                bsc = sc1
            else:
                dash_pair = (lw2, rw2)
                bsc = sc2

            left_word = dash_pair[0]
            right_word = dash_pair[1]

            # That is both words are preferred and one is in lexicon
            # or both words are in the lexicon
            if bsc >= 4:
                tokens_dash.append((left_word, "RWORD"))
                tokens_dash.append((right_word, "RWORD"))
        # end if

        return tokens_dash

    def _tokenize_punctuation(self, tokens: list) -> list:
        tokens4 = []

        for pair in tokens:
            word = pair[0]
            label = pair[1]

            # For dealing with abbreviations
            if label == "PUNCT" and len(word) > 1 and \
                    word.startswith('.') and word != "...":
                tokens4.append((word[0:1], "PUNCT"))
                tokens4.append((word[1:], "PUNCT"))
            else:
                tokens4.append(pair)
            # end if

        return tokens4

    def _tokenize_dashed_words(self, tokens: list) -> list:
        tokens2 = []

        for pair in tokens:
            word = pair[0]

            if '-' in word and not word.startswith('-') and not word.endswith('-'):
                dash_tokens = self._decide_dash_split(word)

                if dash_tokens:
                    tokens2.extend(dash_tokens)
                    continue
                # end if
            # end if

            tokens2.append(pair)
        # end for

        return tokens2

    def _recognize_phrasal_tokens(self, tokens: list, label: str) -> list:
        """Takes a list of tokens and adds MWE or ABBR labels to adjacent
        tokens that form abbreviations or multi-word expressions."""

        tokens3 = []
        i = 0
        reject_phrset = None

        if label == "MWE":
            reject_phrset = RoTokenizer._romanian_reject_mwes
        elif label == "ABBR":
            reject_phrset = RoTokenizer._romanian_reject_abbrs
        # end if

        while i < len(tokens):
            word = tokens[i][0]
            word_is_first = False

            if label == "MWE":
                word_is_first = self._lexicon.is_mwe_first_word(word)
            elif label == "ABBR":
                word_is_first = self._lexicon.is_abbr_first_word(word)
            # end if

            if not word_is_first:
                tokens3.append(tokens[i])
                i += 1
                continue
            # end if

            # Extract the longest phrasal token
            phrtok = [word]
            word_count = 1
            j = i + 1
            max_len = 0

            if label == "MWE":
                max_len = self._lexicon._maxmwelen
            elif label == "ABBR":
                max_len = self._lexicon._maxabbrlen
            # end if

            while word_count < max_len and j < len(tokens):
                word = tokens[j][0]
                tag = tokens[j][1]

                # Do not recognize MWEs over new-lines
                if tag == 'EOL':
                    break
                # end if

                if tag == 'SPACE':
                    if label == "MWE" and phrtok[-1] != '_':
                        phrtok.append('_')
                    # end if
                else:
                    phrtok.append(word)

                    if tag == "RWORD" or tag == "FWORD" or \
                            tag == "WORD" or tag == "ABBR" or word == ".":
                        word_count += 1
                    # end if
                # end if

                j += 1
            # end while

            phrtok_found = False

            # Try all MWEs of different lengths
            for k in range(len(phrtok), 1, -1):
                j = i + k
                phr_word = "".join(phrtok[0:k])

                if (label == "MWE" and phr_word.endswith('_')) or \
                    (label == "ABBR" and not phr_word.endswith('.')):
                    continue
                # end if

                # MWE found. Tag the tokens that compose the MWE.
                if self._lexicon.accept_phrasal_token(phr_word, label) and phr_word.lower() not in reject_phrset:
                    for x in range(i, j):
                        tok = list(tokens[x])
                        tok[1] = label
                        tokens3.append(tuple(tok))

                    i = j
                    phrtok_found = True
                    break
                # end if

            if not phrtok_found:
                tokens3.append(tokens[i])
                i += 1
            # end if
        # end while

        return tokens3

    def tokenize(self, input_string: str) -> list:
        """Takes a Python input_string representing a Romanian text
        and it splits it in words and non-words. This is the main method
        of this class."""
        crt_word = ""
        tokens = []
        last_set_index = -1
        uni_char_sets = RoTokenizer._unicode_char_sets
        uni_char_sets[-1] = set()

        for c in input_string:
            # Do not accept tabs.
            # We use those as delimiters.
            if c == '\t':
                c = ' '
            # end if

            c_cat = uc.category(c)

            for si in range(len(uni_char_sets)):
                # If c is not in any set, save a default set for it.
                if si == len(uni_char_sets) - 1:
                    uni_char_sets[si].add(c)
                # end if

                if last_set_index == -1:
                    if c in uni_char_sets[si] or c_cat in uni_char_sets[si]:
                        crt_word += c
                        last_set_index = si
                        break
                    # end if
                else:
                    if c in uni_char_sets[si] or c_cat in uni_char_sets[si]:
                        if last_set_index == si:
                            crt_word += c
                        else:
                            tokens.append((crt_word, self.tag_word(crt_word)))
                            crt_word = c
                            last_set_index = si
                        # end if
                        break
                # end if
            # end for
        # end for all chars
        
        if crt_word:
            tokens.append((crt_word, self.tag_word(crt_word)))
        # end if

        tokens = self._tokenize_punctuation(tokens)
        tokens = self._tokenize_dashed_words(tokens)
        tokens = self._recognize_phrasal_tokens(tokens, "ABBR")
        tokens = self._recognize_phrasal_tokens(tokens, "MWE")

        return tokens

    def tokenize_file(self, input_file: str) -> list:
        """Takes the input_file (text file, UTF-8 encoded),
        tokenizes it and writers the output to input_file.tok.
        Also returns the list of tokens to the caller."""

        print(stack()[0][3] + ": tokenizing file {0!s}".format(
            input_file), file=sys.stderr, flush=True)

        return self.tokenize(txt_file_to_string(input_file))
