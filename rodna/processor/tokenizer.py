from typing import List, Tuple, Dict
import re
import numpy as np
import unicodedata as uc
from .lexicon import Lex
from ..utils.datafile import txt_file_to_string
from .. import logger


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
    _romanian_punct_chars = set(",.?!\"’´`‘':;()[]{}…„”“«»/-_•●·")
    _romanian_sym_chars = set("<>~@#%^&*+=÷$\\|§©°")
    # \t is reserved. Replace it with ' '
    _romanian_spc_chars = set(" ")
    _romanian_eol_chars = set("\r\n")
    # When splitting the dashed words, e.g. m-ai, prefer these tokens.
    _romanian_dash_split_must = {
        'mi-i': ['mi', '-i'],
        'de-i': ['de', '-i'],
        'c-un': ['c-', 'un'],
        'de-un': ['de-', 'un'],
        'și-un': ['și', '-un'],
        'și-i': ['și', '-i'],
        'și-am': ['și', '-am'],
        'și-ai': ['și', '-ai'],
        'și-ați': ['și', '-ați'],
        'într-una': ['într-', 'una'],
        'și-s': ['și', '-s'],
        'V-lea': ['V', '-lea'],
        'VI-lea': ['VI', '-lea'],
        'VI-a': ['VI', '-a'],
        'ni-i': ['ni', '-i'],
        'de-aș': ['de', '-aș'],
        'de-ar': ['de', '-ar'],
        'de-am': ['de', '-am'],
        'de-ați': ['de', '-ați'],
        'de-ți': ['de', '-ți'],
        'te-năbuși': ['te', '-năbuși']
    }
    _romanian_reject_mwes = set(["de_a"])
    _romanian_reject_abbrs = set()
    # Plus "JUNK" if nothing else matches.
    # SPACE is used in MWE recognition!
    # EOL comes before SPACE (SPACE contains EOL)!
    _token_classes = [
        "ABBR", "NUM", "RWORD", "MWE",
        "FWORD", "WORD", "EOL", "SPACE",
        "PUNCT", "SYM",
        # Some NER classes that appear in RRT
        "DATE", "PERC", "REALNUM", "BULLET",
        "UNKABBR", "ABBRYEAR", "PARREF", "PERSNAME",
        "TIME"]
    _roman_numerals = set([
        "I", "II", "III", "IV", "V",
        "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV",
        "XVI", "XVII", "XVIII", "XIX", "XX",
        "XXI", "XXII", "XXIII", "XXIV", "XXV",
        "XXVI", "XXVII", "XXVIII", "XXIX", "XXX"])
    _number_pattern = re.compile("^\\d+$")
    _special_pattern = re.compile("^[_-]+$")
    _all_uppercase_dash_word = re.compile('^[A-ZȘȚĂÎÂ]+-[A-ZȘȚĂÎÂ]+$')
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
    _named_entities_def: Dict[str, List[List[re.Pattern]]] = {
        'DATE': [
            [
                re.compile('0?[123456789]|1[0-9]|2[0-9]|3[01]'),
                re.compile('[.]'),
                re.compile('0?[123456789]|1[012]'),
                re.compile('[.]'),
                re.compile('(1[0-9]|2[0-9])\\d{2}')
            ],
            [
                re.compile('0?[123456789]|1[0-9]|2[0-9]|3[01]'),
                re.compile('/'),
                re.compile('0?[123456789]|1[012]'),
                re.compile('/'),
                re.compile('(1[0-9]|2[0-9])\\d{2}')
            ],
            [
                re.compile('0?[123456789]|1[0-9]|2[0-9]|3[01]'),
                re.compile('-'),
                re.compile('0?[123456789]|1[012]'),
                re.compile('-'),
                re.compile('(1[0-9]|2[0-9])\\d{2}')
            ]
        ],
        'PERC': [
            [re.compile('\\d+'), re.compile('%')],
            [re.compile('\\d+'), re.compile('[,.]'), re.compile('\\d+'), re.compile('%')]
        ],
        'REALNUM': [
            [re.compile('\\d+'), re.compile('[,.]'), re.compile('\\d+')]
        ],
        'BULLET': [
            [re.compile('\\d+'), re.compile('[.]'), re.compile('\\d+')],
            [re.compile('\\d+'), re.compile('[.]')],
            [re.compile('[a-z0-9]'), re.compile('[.;)]')]
        ],
        'PARREF': [
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]')],
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+')],
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]')],
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+')],
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]')],
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+')],
            [re.compile('\\d+'), re.compile('[.]'),
                re.compile('\\d+'), re.compile('[.]')]
        ],
        'UNKABBR': [
            [re.compile('[A-ZȘȚÎ]'), re.compile('[.]'),
             re.compile('[A-ZȘȚÎ]'), re.compile('[.]'),
             re.compile('[A-ZȘȚÎ]'), re.compile('[.]'),
             re.compile('[A-ZȘȚÎ]'), re.compile('[.]')],
            [re.compile('[A-ZȘȚÎ]'), re.compile('[.]'),
             re.compile('[A-ZȘȚÎ]'), re.compile('[.]'),
             re.compile('[A-ZȘȚÎ]'), re.compile('[.]')],
            [re.compile('[A-ZȘȚÎ]'), re.compile('[.]'),
             re.compile('[A-ZȘȚÎ]'), re.compile('[.]')],
            [re.compile('[A-ZȘȚÎ]'), re.compile('[.]')]
        ],
        'ABBRYEAR': [
            [re.compile("[’´`‘']"), re.compile('[3456789]\\d')]
        ],
        'PERSNAME': [
            [re.compile('O'), re.compile("'"), re.compile('Brien')]
        ],
        'TIME': [
            [re.compile('0?\\d|1\\d|2[0123]'), re.compile(':'), re.compile('[012345]\\d')]
        ]
    }
    _named_entities_msd: Dict[str, str] = {
        'DATE': 'Mc-s-b',
        'BULLET': 'Mc-s-b',
        'PARREF': 'Mc-s-b',
        'PERC': 'Mc-p-d',
        'REALNUM': 'Mc-s-d',
        'UNKABBR': 'Yn',
        'ABBRYEAR': 'Mc-s-d',
        'PERSNAME': 'Np',
        'TIME': 'Mc-p-d'
    }

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

        if self._lexicon.is_lex_word(word, dash_match=True) or \
                self._lexicon.is_lex_word(word.lower(), dash_match=True):
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

        if word in ['i', 'vi', 'vii', 'viii']:
            # Valid Romanian words
            return False
        # end if

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

    def is_time(self, word: str) -> bool:
        return False

    def is_persname(self, word: str) -> bool:
        return False

    def is_realnum(self, word: str) -> bool:
        """No single word can be a real number (e.g. 1,23).
        These labels are assigned after the tokenization runs."""

        return False

    def is_bullet(self, word: str) -> bool:
        """No single word can be a bullet (e.g. 1,23 or 1.2.3.4.).
        These labels are assigned after the tokenization runs."""

        return False

    def is_parref(self, word: str) -> bool:
        return False

    def is_unkabbr(self, word: str) -> bool:
        return False
    
    def is_abbryear(self, word: str) -> bool:
        return False

    def is_date(self, word: str) -> bool:
        """No single word can be a date (e.g. 25.12.2026).
        These labels are assigned after the tokenization runs."""

        return False
    
    def is_perc(self, word: str) -> bool:
        """No single word can be a percent (e.g. 25%).
        These labels are assigned after the tokenization runs."""

        return False

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

        return False

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

    def _decide_two_word_dash_split(self, left_word: str, right_word: str) -> List[Tuple[str, str]]:
        # 1. Honor defined splits
        left_word_dash = left_word + '-'
        dash_right_word = '-' + right_word
        full_word = left_word + '-' + right_word

        # 1.1 Incoming case
        if full_word in RoTokenizer._romanian_dash_split_must:
            imposed_split = RoTokenizer._romanian_dash_split_must[full_word]

            if left_word_dash == imposed_split[0]:
                return [
                    (left_word_dash, "RWORD"),
                    (right_word, "RWORD")
                ]
            elif dash_right_word == imposed_split[1]:
                return [
                    (left_word, "RWORD"),
                    (dash_right_word, "RWORD")
                ]
            # end if
        # end if

        full_word_lc = full_word.lower()

        # 1.2 Lower-case
        if full_word_lc in RoTokenizer._romanian_dash_split_must:
            imposed_split = RoTokenizer._romanian_dash_split_must[full_word_lc]

            if left_word_dash.lower() == imposed_split[0]:
                return [
                    (left_word_dash, "RWORD"),
                    (right_word, "RWORD")
                ]
            elif dash_right_word.lower() == imposed_split[1]:
                return [
                    (left_word, "RWORD"),
                    (dash_right_word, "RWORD")
                ]
            # end if
        # end if

        # 2. Do not split e.g. 'e-mail', 'după-amiază', etc.
        if self._lexicon.is_lex_word(word=full_word):
            return []
        # end if

        # 3. Check if split is impossible
        if self._lexicon.is_clitic_word(left_word_dash) == -1 and \
                self._lexicon.is_clitic_word(dash_right_word) == -1 and \
                not self._lexicon.can_be_func_word(word=left_word) and \
                not self._lexicon.can_be_func_word(word=right_word):
            if self._lexicon.is_lex_word(word=left_word) and \
                    self._lexicon.is_lex_word(word=right_word):
                # aducerile-aminte, șobolanul-pierdut
                return [
                    (left_word_dash, "RWORD"),
                    (right_word, "RWORD")
                ]
            elif (dash_right_word.startswith('-n') or \
                    dash_right_word.startswith('-m')) and \
                    not self._lexicon.is_lex_word(word=right_word):
                # Unknown -n, -m words, e.g. -ncuietoarea
                return [
                    (left_word, "RWORD"),
                    (dash_right_word, "RWORD")
                ]
            else:
                return []
            # end if
        # end if

        # 4. Check hard-coded cases
        if left_word.lower() == 'și' and \
                self._lexicon.is_pos_word(right_word, pos='RANV') and \
                not self._lexicon.can_be_func_word(dash_right_word):
            # și-atunci, și-acum, și-ochioasa
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        # end if

        # 5. If right_word is capitalized, skip the split.
        # It is more probable that this is a name containing a dash.
        # Examples: 'Al-Qaida', 'L-Dopa', C45-Q43
        if (right_word[0].lower() != right_word[0] or \
                right_word[0] in '0123456789') and \
                not RoTokenizer._all_uppercase_dash_word.fullmatch(full_word):
            return []
        # end if

        # 6. See if 'left_word_dash right_word' is necessary.
        if (self._lexicon.is_pos_word(left_word, pos='S') or
                self._lexicon.is_pos_word(left_word_dash, pos='S')) and \
                (self._lexicon.can_be_pos_word(right_word, pos='TD') or
                 self._lexicon.is_pos_word(right_word, pos='R')):
            # peste-al, sub-a, pe-ale, etc.
            return [
                (left_word_dash, "RWORD"),
                (right_word, "RWORD")
            ]
        # end if

        # 7. See if 'left_word dash_right_word' is necessary.
        if self._lexicon.is_clitic_word(left_word) == 1 or \
            (self._lexicon.is_clitic_word(left_word_dash) == -1 and \
                self._lexicon.is_clitic_word(dash_right_word) > -1):
            # not self._lexicon.can_be_pos_word(left_word, pos='S')
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        elif self._lexicon.is_clitic_word(dash_right_word) == 1 and \
                self._lexicon.is_pos_word(dash_right_word, pos='S'):
            # e-mpotriva
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        elif self._lexicon.is_clitic_word(dash_right_word) == 1 and \
                self._lexicon.is_pos_word(dash_right_word, pos='VNAR') and \
                not self._lexicon.can_be_func_word(dash_right_word):
            # se-ntinde, și-ntinderea, și-ntinsa
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        elif self._lexicon.is_clitic_word(dash_right_word) == 1 and \
                self._lexicon.is_pos_word(dash_right_word, pos='T') and \
                self._lexicon.is_pos_word(left_word, pos='M', exact_match=True):
            # V-lea, I-ul, VI-lea
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        elif self._lexicon.is_clitic_word(left_word_dash) == 0 and \
                self._lexicon.is_clitic_word(dash_right_word) == 0 and \
                self._lexicon.is_pos_word(dash_right_word, pos='PRSD'):
            # și-l, i-l, și-mi, etc.
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        elif self._lexicon.is_clitic_word(dash_right_word) == 1 and \
                self._lexicon.is_clitic_word(left_word_dash) == 1 and \
                not self._lexicon.is_pos_word(left_word_dash, pos='S') and \
                (
                    not self._lexicon.is_lex_word(right_word) or
                    self._lexicon.is_pos_word(left_word, pos='V') or
                    self._lexicon.is_pos_word(dash_right_word, pos='PDT') or
                    self._lexicon.is_pos_word(dash_right_word, pos='S') or
                    self._lexicon.is_pos_word(dash_right_word, pos='R') or
                    self._lexicon.is_pos_word(dash_right_word, pos='DR')
                ):
            # 7.1 It's a tie: both left and right attachments are equally likely.
            # Only attach to right if left word is in dictionary and cannot be a clitic
            # without a dash
            return [
                (left_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        else:
            # 8. Default split: 'left_word_dash right_word'
            return [
                (left_word_dash, "RWORD"),
                (right_word, "RWORD")
            ]
        # end if

    def _decide_three_word_dash_split(self, left_word: str, middle_word: str, right_word: str) -> List[Tuple[str, str]]:
        def first_letter(word: str) -> bool:
            return word[0] != word[0].lower() or word[0] in '0123456789'
        # end def

        # Do not split names of different things
        if first_letter(word=middle_word) or \
                first_letter(word=right_word):
            return []
        # end if

        full_word = left_word + '-' + middle_word + '-' + right_word

        # If the word is in lexicon, leave it alone
        if self._lexicon.is_lex_word(word=full_word):
            return []
        # end if

        left_middle_word = left_word + '-' + middle_word
        dash_right_word = '-' + right_word

        # e-mail-ul
        if self._lexicon.is_clitic_word(dash_right_word) >= 0 and \
                self._lexicon.is_lex_word(word=left_middle_word):
            return [
                (left_middle_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        # end if

        left_word_dash = left_word + '-'
        middle_right_word = middle_word + '-' + right_word

        if self._lexicon.is_clitic_word(left_word_dash) >= 0 and \
                self._lexicon.is_lex_word(word=middle_right_word):
            return [
                (left_word_dash, "RWORD"),
                (middle_right_word, "RWORD")
            ]
        # end if

        dash_middle_word = '-' + middle_word

        # Most frequent case: dându-li-se, mânca-mi-ați, etc.
        # ca-ntr-o, și-ntr-o
        if self._lexicon.is_pos_word(left_word, pos='V') or \
                self._lexicon.can_be_pos_word(left_word, pos='R') or \
                self._lexicon.can_be_pos_word(left_word, pos='C'):
            if (self._lexicon.is_clitic_word(dash_middle_word) >= 0 or
                    self._lexicon.can_be_func_word(dash_middle_word)) and \
                    (self._lexicon.is_clitic_word(dash_right_word) >= 0 or
                     self._lexicon.can_be_func_word(dash_right_word)):
                # Only if parts can be clitics or functional words
                return [
                    (left_word, "RWORD"),
                    (dash_middle_word, "RWORD"),
                    (dash_right_word, "RWORD")
                ]
            # end if
        # end if

        # ți-ar-ncepe
        if self._lexicon.is_pos_word(dash_right_word, pos='V') and \
                self._lexicon.is_clitic_word(dash_right_word) >= 0:
            result = self._decide_two_word_dash_split(left_word, middle_word)
            result.append((dash_right_word, 'RWORD'))
            return result
        # end if

        # te-ncrezi-n
        if self._lexicon.is_pos_word(dash_middle_word, pos='V') and \
                self._lexicon.is_clitic_word(dash_middle_word) >= 0 and \
                self._lexicon.is_clitic_word(dash_right_word) >= 0:
            return [
                (left_word, "RWORD"),
                (dash_middle_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        # end if

        # Iată-ne-alături
        if self._lexicon.is_pos_word(left_word, pos='I') and \
                self._lexicon.is_clitic_word(dash_middle_word) >= 0:
            return [
                (left_word, "RWORD"),
                (dash_middle_word, "RWORD"),
                (dash_right_word, "RWORD")
            ]
        # end if
        
        return []

    def _decide_dash_split(self, word: str) -> List[Tuple[str, str]]:
        tokens_dash = []
        word_parts = word.split('-')

        if len(word_parts) == 3:
            tokens_dash = self._decide_three_word_dash_split(left_word=word_parts[0],
                                                             middle_word=word_parts[1],
                                                             right_word=word_parts[2])
        # end if

        if len(word_parts) == 2:
            tokens_dash = self._decide_two_word_dash_split(left_word=word_parts[0],
                                                           right_word=word_parts[1])
        # end if

        # Re-classify tokens here
        return [(w, self.tag_word(word=w)) for w, _ in tokens_dash]

    def _tokenize_punctuation(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
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

    def _tokenize_dashed_words(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
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

    def _recognize_named_entities(self, tokens: List[Tuple[str, str]], label: str) -> List[Tuple[str, str]]:
        i = 0
        tokens2 = []

        while i < len(tokens):
            tok, lbl = tokens[i]

            if lbl in RoTokenizer._named_entities_def:
                # Do not overwrite what a previous match did
                tokens2.append(tokens[i])
                i += 1
            else:
                ner_found = False

                # Try all regex sequences for the NER label
                for rx_seq in RoTokenizer._named_entities_def[label]:
                    j = i
                    tok, _ = tokens[j]
                    ner_found = True

                    # Token and regex match, token by token
                    for k in range(len(rx_seq)):
                        if not rx_seq[k].fullmatch(tok):
                            ner_found = False
                            break
                        # end if

                        j += 1

                        if j == len(tokens):
                            if k < len(rx_seq) - 1:
                                ner_found = False
                            # end if

                            break
                        # end if

                        tok, _ = tokens[j]
                    # end for rx in sequence

                    if ner_found and label == 'BULLET' and i > 0:
                        # Only allow BULLET labels at beginning
                        # of the sentence.
                        ner_found = False
                    # end if

                    if ner_found:
                        for k in range(i, j):
                            tokens2.append((tokens[k][0], label))
                        # end for

                        i = j
                        break
                    # end if
                # end for

                if not ner_found:
                    tokens2.append(tokens[i])
                    i += 1
                # end if
            # end if
        # end i

        return tokens2

    def _recognize_phrasal_tokens(self, tokens: List[Tuple[str, str]], label: str) -> List[Tuple[str, str]]:
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
                # max_len is in number of parts, e.g. P.S._Sa => len is 2
                max_len = self._lexicon._maxmwelen
            elif label == "ABBR":
                # max_len is in number of parts, e.g. O.N.U. => len is 3
                max_len = self._lexicon._maxabbrlen
            # end if

            while word_count < max_len and j < len(tokens):
                word = tokens[j][0]
                tag = tokens[j][1]

                # Do not recognize MWEs over new-lines
                # Do not recognize ABBRs over spaces
                if tag == 'EOL' or \
                        (tag == 'SPACE' and label == 'ABBR'):
                    break
                # end if

                if tag == 'SPACE':
                    if label == "MWE" and phrtok[-1] != '_':
                        phrtok.append('_')
                    # end if
                else:
                    phrtok.append(word)

                    if tag == "RWORD" or tag == "FWORD" or \
                            tag == "WORD" or tag == "ABBR":
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

    def ne_label_to_msd(self, label: str) -> str:
        if label in RoTokenizer._named_entities_msd:
            RoTokenizer._named_entities_msd[label]
        else:
            return ''
        # end if

    def tokenize(self, input_string: str) -> List[Tuple[str, str]]:
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
        # C.O.R.
        tokens = self._recognize_phrasal_tokens(tokens, label="ABBR")
        # împreună_cu
        tokens = self._recognize_phrasal_tokens(tokens, label="MWE")
        # 21.10.2026
        tokens = self._recognize_named_entities(tokens, label="DATE")
        # 7,21% or 100%
        tokens = self._recognize_named_entities(tokens, label="PERC")
        # 1.2.3.4. or 1.2.3
        tokens = self._recognize_named_entities(tokens, label="PARREF")
        # e. or 2., only at the beginning of the sentence
        tokens = self._recognize_named_entities(tokens, label="BULLET")
        # 7.1 or 8,23
        tokens = self._recognize_named_entities(tokens, label="REALNUM")
        # S.U.A.
        tokens = self._recognize_named_entities(tokens, label="UNKABBR")
        # '70
        tokens = self._recognize_named_entities(tokens, label="ABBRYEAR")
        # O'Brien
        tokens = self._recognize_named_entities(tokens, label="PERSNAME")
        # 7:30
        tokens = self._recognize_named_entities(tokens, label="TIME")

        return tokens

    def tokenize_file(self, input_file: str) -> List[Tuple[str, str]]:
        """Takes the input_file (text file, UTF-8 encoded),
        tokenizes it and writers the output to input_file.tok.
        Also returns the list of tokens to the caller."""

        logger.info(f"Tokenizing file [{input_file}]")

        return self.tokenize(txt_file_to_string(input_file))

    def glue_tokens(self, tokens: List[Tuple[str, str]], do_mwes: bool = False) -> List[Tuple[str, str]]:
        """Glues some tokens such as abbreviations, NEs or MWEs in single tokens, for downstream processing.
        If `do_mwes is True`, multiword expressions are also glued.
        Also, splits punctuation and symbols, one char per token.
        Finally, ensure that final '.' if glued to the previous token is unglued."""

        if not tokens:
            return tokens
        # end if

        glued_tokens: List[Tuple[str, str]] = []
        expr_tokens = []
        expr_label = ''

        for token, tlabel in tokens:
            if tlabel == 'ABBR' or tlabel in RoTokenizer._named_entities_def or \
                    (do_mwes and tlabel == 'MWE'):
                expr_tokens.append(token)
                expr_label = tlabel
            else:
                if expr_tokens:
                    expr_token = ''.join(expr_tokens)
                    expr_token = expr_token.replace(' ', '_')
                    glued_tokens.append((expr_token, expr_label))
                    expr_tokens = []
                    expr_label = ''
                # end if

                if (tlabel == 'PUNCT' or tlabel == 'SYM') and \
                        len(token) > 1:
                    glued_tokens.extend([(pstk, tlabel) for pstk in token])
                else:
                    glued_tokens.append((token, tlabel))
                # end if
            # end if
        # end for

        if expr_tokens:
            expr_token = ''.join(expr_tokens)
            expr_token = expr_token.replace(' ', '_')
            glued_tokens.append((expr_token, expr_label))
        # end if

        # Make sure that if sentence ends with an abbreviation
        # separate the last dot and make it a single token, if
        # word is ambiguous, e.g. 'alin.' or 'loc.' or 'tel.'
        last_word = glued_tokens[-1][0]
        last_label = glued_tokens[-1][1]

        if last_word.endswith('.') and len(last_word) > 1:
            if self._lexicon.abbr_ambiguous_word(last_word) and \
                    last_label == 'ABBR':
                glued_tokens[-1] = (last_word[:-1], 'RWORD')
                glued_tokens.append(('.', 'PUNCT'))
            elif last_label == 'PARREF' or \
                    (last_label == 'UNKABBR' and len(last_word) == 2):
                glued_tokens[-1] = (last_word[:-1], last_label)
                glued_tokens.append(('.', 'PUNCT'))
            # end if
        # end if

        return glued_tokens
