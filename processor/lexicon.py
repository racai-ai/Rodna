import re
from typing import Pattern, Set, List, Dict
import numpy as np
from . import logger, log_once, logging, \
    TBL_WORDFORM_FILE, MSD_MAP_FILE, MORPHO_MAP_FILE


class MSD(object):
    """This class will model a Morpho-Syntactic Descriptor.
    It will return a real-valued, fixed-length vector for a MSD."""

    punct_ctag = 'PUNCT'
    content_word_ctag_pattern = re.compile("^(A[SPN]?|M|N[NPS]|V[123NP]|Y|R)")
    # Both MSD and CTAG have this
    unknown_label = 'X'
    unknown_punct_msd = 'Z'
    number_msd = 'Mc-s-d'
    bullet_number_msd = 'Mc-s-b'
    punct_msd_inventory = {
        ".": "Zp",
        ",": "Zc",
        ":": "Zl",
        ";": "Zs",
        "-": "Zd",
        "?": "Zq",
        "!": "Zx",
        "...": "Zh",
        "…": "Zh",
        "/": "Zo",
        "_": "Zu",
        # Pair tags
        # Z1 = open punctuation
        # Z2 = close punctuation
        "(": "Z1p",
        "[": "Z1p",
        "{": "Z1p",
        ")": "Z2p",
        "]": "Z2p",
        "}": "Z2p",
        # Single/double quotes
        "\"": "Z1q",
        "``": "Z1q",
        "''": "Z2q",
        "'": "Z1q",
        "„": "Z1q",
        "”": "Z2q",
        "“": "Z1q",
        "«": "Z1q",
        "»": "Z2q"
    }
    # When writing CoNLL-U files, use these
    # tags instead of the newly defined MSDs above.
    punct_ctag_inventory = {
        '-': 'DASH',
        '−': 'DASH',
        '–': 'DASH',
        '—': 'DASH',
        '―': 'DASH',
        '\'': 'QUOT',
        '!': 'EXCL',
        '!...': 'EXCLHELLIP',
        '"': 'DBLQ',
        '%': 'PERCENT',
        '(': 'LPAR',
        ')': 'RPAR',
        '*': 'STAR',
        ',': 'COMMA',
        '.': 'PERIOD',
        '...': 'HELLIP',
        '/': 'SLASH',
        ':': 'COLON',
        ';': 'SCOLON',
        '?': 'QUEST',
        '[': 'LSQR',
        ']': 'RSQR',
        '_': 'UNDERSC',
        '{': 'LCURL',
        '}': 'RCURL',
        '’': 'QUOT',
        '‚': 'QUOT',
        '“': 'DBLQ',
        '”': 'DBLQ',
        '„': 'DBLQ',
        '+': 'PLUS',
        '<': 'LT',
        '=': 'EQUAL',
        '>': 'GT',
        '±': 'PLUSMINUS',
        '«': 'DBLQ',
        '»': 'DBLQ',
        '≥': 'GE',
        '→': 'ARROW',
        '…': 'HELLIP',
        '•': 'BULLET'
    }
    punct_patt = re.compile("^\\W+$")
    _msd_attributes_categories = [
        ("NounType", ("c", "p")),
        ("VerbType", ("m", "a", "o", "c")),
        ("AdjType", ("f")),
        ("PronType", ("p", "d", "i", "s", "x", "z", "w")),
        ("DetType", ("d", "i", "s", "z", "w", "h")),
        ("ArtType", ("f", "d", "i", "s")),
        ("AdvType", ("g", "p", "z", "m", "w", "c")),
        ("AdpType", ("p")),
        ("ConjType", ("c", "s", "r")),
        ("NumType", ("c", "o", "f", "m", "l")),
        ("PartType", ("z", "n", "s", "a", "f")),
        ("AbbrevType", ("n", "v", "a", "r", "p")),
        ("PuncType", ("c", "p", "l", "s", "d", "q", "x", "h", "o", "u", "1", "2")),
        ("PuncPairType", ("p", "q")),
        ("EndType", ("b", "e")),
        # General attributes, applicable to all POSes
        ("Gender", ("m", "f", "n")),
        ("Number", ("s", "p")),
        ("OwnNumber", ("s", "p")),
        ("Case", ("v", "r", "o", "n", "a", "d", "g")),
        ("Definiteness", ("n", "y")),
        ("Clitic", ("n", "y")),
        ("VerbForm", ("i", "s", "m", "n", "p", "g")),
        ("Tense", ("p", "i", "s", "l")),
        ("Person", ("1", "2", "3")),
        ("Degree", ("p", "c", "s")),
        ("PronForm", ("s", "w")),
        ("ModifierType", ("e", "o")),
        ("WordFormation", ("s", "c")),
        ("CoordType", ("s", "r", "c")),
        ("SubordType", ("z", "p")),
        ("NumForm", ("d", "r", "l", "b"))
    ]
    _msd_by_category = {
        "N": ("NounType", "Gender", "Number", "Case", "Definiteness", "Clitic"),
        "V": ("VerbType", "VerbForm", "Tense", "Person", "Number", "Gender", "-", "-", "-", "Clitic"),
        "A": ("AdjType", "Degree", "Gender", "Number", "Case", "Definiteness", "Clitic"),
        "P": ("PronType", "Person", "Gender", "Number", "Case", "OwnNumber", "-", "Clitic", "-", "-", "-", "-", "-", "PronForm"),
        "D": ("DetType", "Person", "Gender", "Number", "Case", "OwnNumber", "-", "Clitic", "ModifierType"),
        "T": ("ArtType", "Gender", "Number", "Case", "Clitic"),
        "R": ("AdvType", "Degree", "Clitic"),
        "S": ("AdpType", "WordFormation", "Case", "Clitic"),
        "C": ("ConjType", "WordFormation", "CoordType", "SubordType", "Clitic"),
        "M": ("NumType", "Gender", "Number", "Case", "NumForm", "Definiteness", "Clitic"),
        "Q": ("PartType", "-", "Clitic"),
        "Y": ("AbbrevType", "Gender", "Number", "Case", "Definiteness"),
        "X": (),
        "Z": ("PuncType", "PuncPairType"),
        # These are the sentence START/END MSDs.
        "L": ("EndType")
    }

    def __init__(self):
        """Takes the MSD -> CTAG map file, commonly named msdtag.ro.map"""
        # This is the dictionary of all possible MSDs
        self._msdinventory = {}
        self._msdinverseinv = []
        # Will store the length of the output MSD vector
        self._msdoutputsize = 0
        # On which position in the self._vectorAttrs the X MSD is
        # This MSD is returned for any invalid MSD requests
        self._xattrindex = -1
        self._xindex = -1
        # The size of the real-valued vector that represents a MSD at input
        # That is, the values of the MSD's attributes, stored at the appropriate indexes.
        self._msdinputsize = len(MSD._msd_by_category)
        self._msd_poses = sorted(list(MSD._msd_by_category.keys()))

        for (_, attr_vals) in MSD._msd_attributes_categories:
            self._msdinputsize += len(attr_vals)
        # end for

        # CTAG data structures
        self._msdtoctag = {}
        self._ctagtomsd = {}
        self._ctaginventory = {}
        self._ctaginverseinv = []
        self._ctagoutputsize = 0
        self._ctagxindex = -1
        self._create_msd_inventory(MSD_MAP_FILE)
        self._msdtomorpho = self._create_morpho_feats_inventory(
            MORPHO_MAP_FILE)

    @staticmethod
    def get_msd_pos(msd: str) -> str:
        if msd:
            return msd[0]
        else:
            return '?'
        # end if

    @staticmethod
    def get_start_end_tags(limit: str, ctag: bool = True) -> str:
        """If `ctag` is `False`, get the MSD start/end tags."""

        if limit.lower() == 'start' or \
                limit.lower() == 'begin' or \
                limit.lower() == 'beginning':
            if ctag:
                return 'SBEG'
            else:
                return 'Lb'
            # end if
        elif limit.lower() == 'end':
            if ctag:
                return 'SEND'
            else:
                return 'Le'
            # end if
        # end if

        return ''

    def get_ctag_inventory(self) -> dict:
        return self._ctaginventory

    def get_input_vector_size(self):
        return self._msdinputsize

    def get_ctag_input_vector_size(self):
        """Same as output as this is a sum of one-hot vectors
        of possible CTAGs for the input word."""
        return self._ctagoutputsize

    def get_output_vector_size(self):
        return self._msdoutputsize

    def get_ctag_output_vector_size(self):
        return self._ctagoutputsize

    def get_punct_ctag(self, punct_token: str) -> str:
        if punct_token in MSD.punct_ctag_inventory:
            return MSD.punct_ctag_inventory[punct_token]
        else:
            return MSD.punct_ctag
        # end if

    def idx_to_msd(self, index: int) -> str:
        if index < 0 or index >= len(self._msdinverseinv):
            return 'Unk'
        else:
            return self._msdinverseinv[index]
        # end if

    def idx_to_ctag(self, index: int) -> str:
        if index < 0 or index >= len(self._ctaginverseinv):
            return 'Unk'
        else:
            return self._ctaginverseinv[index]
        # end if

    def msd_to_idx(self, msd: str) -> int:
        if msd in self._msdinventory:
            return self._msdinventory[msd]
        else:
            return self._xindex
        # end if

    def get_x_idx(self) -> int:
        return self._xindex

    def ctag_to_idx(self, ctag: str) -> int:
        if ctag in self._ctaginventory:
            return self._ctaginventory[ctag]
        else:
            return self._ctagxindex
        # end if

    def msd_to_ctag(self, msd: str) -> str:
        """This is in a 1:1 relationship."""

        if msd in self._msdtoctag:
            return self._msdtoctag[msd]
        else:
            return 'Unk'

    def msd_to_upos(self, msd: str, tok_tag: str) -> str:
        """Gets the UD UPOS tag from the MSD. Deterministic mapping.
        - `tok_tag` is the tokenizer tag
        - `msd` is the MSD"""

        if tok_tag == 'PUNCT' or tok_tag == 'SYM':
            return tok_tag
        # end if

        if msd.startswith('Z'):
            return 'PUNCT'
        # end if

        if re.match('^Np', msd):
            return 'PROPN'
        elif re.match('^(Nc|Yn|Y)', msd):
            return 'NOUN'
        elif re.match('^M', msd):
            return 'NUM'
        elif re.match('^(A|Ya)', msd):
            return 'ADJ'
        elif re.match('^(R|Yr)', msd):
            return 'ADV'
        elif re.match('^I', msd):
            return "INTJ"
        elif re.match('^S', msd):
            return "ADP"
        elif re.match('^C[cr]', msd):
            return "CCONJ"
        elif re.match('^Cs', msd):
            return "SCONJ"
        elif re.match('^[DT]', msd):
            return "DET"
        elif re.match('^(P|Yp)', msd):
            return "PRON"
        elif re.match('^(Vm|Yv)', msd):
            return "VERB"
        elif re.match('^Va', msd):
            return "AUX"
        elif re.match('^Q', msd):
            return "PART"
        # end if

        return 'X'

    def msd_to_morpho_feats(self, msd: str) -> str:
        if msd in self._msdtomorpho:
            return self._msdtomorpho[msd]
        else:
            return '_'

    def ctag_to_possible_msds(self, ctag: str) -> list:
        """This is in a 1:m relationship."""

        if ctag in self._ctagtomsd:
            return self._ctagtomsd[ctag]
        else:
            return []

    def is_valid_msd(self, msd: str) -> bool:
        return msd in self._msdinventory

    def is_valid_ctag(self, ctag: str) -> bool:
        return ctag in self._ctaginventory

    def _create_msd_inventory(self, mapfile: str):
        """Constructs the vector of possible MSD attributes."""

        # 1. Read all MSDs and put their attributes in positional sets.
        with open(mapfile, mode="r") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                msd = parts[0]
                ctag = parts[1]

                if msd not in self._msdtoctag:
                    self._msdtoctag[msd] = ctag
                # end if

                if ctag not in self._ctagtomsd:
                    self._ctagtomsd[ctag] = [msd]
                else:
                    self._ctagtomsd[ctag].append(msd)
                # end if

                if ctag not in self._ctaginventory:
                    self._ctaginverseinv.append("")
                    self._ctaginventory[ctag] = self._ctagoutputsize
                    self._ctaginverseinv[self._ctagoutputsize] = ctag

                    if ctag == MSD.unknown_label:
                        self._ctagxindex = self._ctagoutputsize
                    # end if

                    self._ctagoutputsize += 1
                # end if

                # Store MSD in inventory, so that we know
                # which are the valid MSDs
                if msd not in self._msdinventory:
                    self._msdinverseinv.append("")
                    self._msdinventory[msd] = self._msdoutputsize
                    self._msdinverseinv[self._msdoutputsize] = msd

                    if msd == MSD.unknown_label:
                        self._xindex = self._msdoutputsize
                    # end if

                    self._msdoutputsize += 1
                # end if
            # end for all lines
        # end with

        # 1.1 Add PUNCT CTAG
        self._ctaginverseinv.append("")
        self._ctaginventory[MSD.punct_ctag] = self._ctagoutputsize
        self._ctaginverseinv[self._ctagoutputsize] = MSD.punct_ctag
        self._ctagoutputsize += 1

        # 1.2 Add Z MSD
        self._msdinverseinv.append("")
        self._msdinventory[MSD.unknown_punct_msd] = self._msdoutputsize
        self._msdinverseinv[self._msdoutputsize] = MSD.unknown_punct_msd
        self._msdoutputsize += 1

        self._msdtoctag[MSD.unknown_punct_msd] = MSD.punct_ctag

        # 2. Add punctuation MSDs as well
        for punct in MSD.punct_msd_inventory:
            msd = MSD.punct_msd_inventory[punct]

            if msd not in self._msdtoctag:
                self._msdtoctag[msd] = MSD.punct_ctag
            # end if

            if MSD.punct_ctag not in self._ctagtomsd:
                self._ctagtomsd[MSD.punct_ctag] = [msd]
            else:
                self._ctagtomsd[MSD.punct_ctag].append(msd)
            # end if

            if msd not in self._msdinventory:
                self._msdinverseinv.append("")
                self._msdinventory[msd] = self._msdoutputsize
                self._msdinverseinv[self._msdoutputsize] = msd
                self._msdoutputsize += 1
            # end if
        # end for

    def _create_morpho_feats_inventory(self, morfile: str) -> Dict[str, str]:
        morpho_feats = {}

        with open(morfile, mode='r', encoding='utf-8') as f:
            for line in f:
                msd, mfeats = line.strip().split()

                if msd not in morpho_feats:
                    morpho_feats[msd] = mfeats
                else:
                    logger.warning(f'MSD [{msd}] is not unique in the morphological features dict')
                # end if
            # end for
        # end with

        return morpho_feats

    def get_x_input_vector(self) -> np.ndarray:
        """Returns the input vector for the MSD 'X'."""

        msd_vec = np.zeros(self._msdinputsize, dtype=np.float32)
        xi = self._msd_poses.index(MSD.unknown_label)
        msd_vec[xi] = 1.0

        return msd_vec

    def get_ctag_x_input_vector(self) -> np.ndarray:
        """Returns the input vector for the CTAG 'X'."""
        return self.ctag_input_vector('X')

    def ctag_input_vector(self, ctag: str) -> np.ndarray:
        """This method takes a str CTAG and returns the numpy
        binary vector representation for it."""

        ctag_vec = np.zeros(self._ctagoutputsize, dtype=np.float32)

        if ctag in self._ctaginventory:
            ci = self._ctaginventory[ctag]
            ctag_vec[ci] = 1.0
        else:
            ctag_vec[self._ctagxindex] = 1.0
        # end if

        return ctag_vec

    def msd_input_vector(self, msd: str) -> np.ndarray:
        """This method takes a str MSD and returns the numpy
        binary vector representation for it."""

        pos = msd[0]

        # If MSD is not in inventory, return the X representation
        if msd not in self._msdinventory or pos not in MSD._msd_by_category:
            return self.get_x_input_vector()
        # end if

        msd_vec = np.zeros(self._msdinputsize, dtype=np.float32)
        msd_categories = MSD._msd_by_category[pos]

        # 1. Set POS
        pi = self._msd_poses.index(pos)
        msd_vec[pi] = 1.0

        # 2 Set MSD attributes
        # The improved part is that general morphologic categories like the 'number'
        # which can be 'singular' and 'plural' go on the same position in the MSD
        # input vector, regardless of POS: so verbs and nouns will have number on the same index,
        # which helps with the building up of morphological information as these vectors are added
        # for all possible MSDs of a word.
        for i in range(1, len(msd)):
            val_i = msd[i]
            type_i = msd_categories[i - 1]

            if val_i != '-' and type_i != '-':
                # Start with the indexing after the part with POSes
                # in the flattened input vector
                j = len(self._msd_poses)
                val_i_set = False

                for (attr_type, attr_values) in MSD._msd_attributes_categories:
                    if attr_type == type_i:
                        for av in attr_values:
                            if av == val_i:
                                msd_vec[j] = 1.0
                                val_i_set = True
                                break
                            else:
                                j += 1
                            # end if
                        # end for
                    else:
                        j += len(attr_values)
                    # end if

                    if val_i_set:
                        break
                    # end if
                # end for all attribute types
            # end if specified MSD attribute, i.e. not '-'
        # end all positions in the MSD

        return msd_vec

    def get_x_reference_vector(self):
        return self.msd_reference_vector(MSD.unknown_label)

    def msd_reference_vector(self, msd: str) -> np.ndarray:
        """This method will return an one-hot vector which
        has the corresponding index set to 1 for the given MSD."""

        msd_vec = np.zeros(self._msdoutputsize, dtype=np.float32)

        if msd not in self._msdinventory:
            msd_vec[self._xindex] = 1.0
        else:
            msd_vec[self._msdinventory[msd]] = 1.0
        # end if

        return msd_vec


class Lex(object):
    """This class will read in the lexicon (in tbl.wordform format)"""

    # Pronouns, Determiners, Particles, Adpositions, Conjunctions, Numerals
    # proper nouns, numerals, adverbs, not general.
    _mwe_pos_pattern = re.compile("^([PDQSCMI]|Np|R[^g])")
    # Abbreviations
    _abbr_pos_pattern = re.compile("^Y")
    content_word_pos_pattern = re.compile("^([YNAM]|Vm|Rg)")
    _comm_pattern = re.compile("^\\s*#")
    _verb_no_number_pattern = re.compile('^Vmi[pisl][123]$')
    # Length of the affixes to do affix analysis.
    _prefix_length = 5
    _suffix_length = 5
    sentence_case_pattern = re.compile("^[A-ZȘȚĂÎÂ][a-zșțăîâ_-]+$")
    mixed_case_pattern = re.compile(
        "^[a-zA-ZșțăîâȘȚĂÎÂ]*[a-zșțăîâ-][A-ZȘȚĂÎÂ][a-zA-ZșțăîâȘȚĂÎÂ-]*$")
    code_case_pattern = re.compile(
        "^[A-Z][A-ZȘȚĂÎÂ-]*\\d[A-Za-z0-9șțăîâȘȚĂÎÂ-]*$")
    _simple_word_pattern = re.compile('^[a-zșțăîâ]+$')
    upper_case_pattern = re.compile("^[A-ZȘȚĂÎÂ_-]+$")
    number_pattern = re.compile("^(\\d+|\\d+[.,]\\d+)$")
    bullet_number_pattern = re.compile("^(.*\\d[./-].+|.*[./-]\\d.*)$")
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
        code_case_pattern,
        # Punctuation
        re.compile("^\\W+$")
    ]

    def __init__(self):
        # Dictionary lexicon
        self._lexicon: Dict[str, Set[str]] = {}
        self._lemma_lexicon: Dict[str, Dict[str, List[str]]] = {}
        # The MSD representation
        self._msd = MSD()
        # Possible POSes for each word in the lexicon
        self._possibletags: Dict[str, int] = {"UNK": 0}
        self._tagid = 1
        # Maximum length of a multi-word expression (MWE)
        self._maxmwelen = 2
        self._maxabbrlen = 2
        self._mwefirstword: Set[str] = set()
        self._abbrfirstword: Set[str] = set()
        # The set of 'a fi' word forms, lower-cased
        self._tobewordforms: Set[str] = set()
        self._canwordforms: Set[str] = set()
        # 'canioane' has lemma 'canion', so we need to learn
        # to change the word root
        # Also 'băiatul' vs. 'băieții'
        self._inflectional_class: Dict[str, Dict[str, List[str]]] = {
            'noun': {},
            'adje_masc': {},
            'adje_fem': {},
            'verb': {}
        }
        # For abbreviations with 2 tokens, e.g. 'etc.', 'nr.', etc.
        # They have effectively one token
        self._abbrfirstword1: Set[str] = set()
        self.longestwordlen = 20
        self._prefixes: Dict[str, Dict[str, int]] = {}
        self._suffixes: Dict[str, Dict[str, int]] = {}
        self._read_tbl_wordform()
        self._remove_abbr_first_words_that_are_lex_words()

    def get_msd_object(self) -> MSD:
        return self._msd

    def get_inflectional_classes(self) -> Dict[str, Dict[str, List[str]]]:
        """Returns the lemma to its possible inflectional forms dictionary."""

        return self._inflectional_class

    def get_lemma_lexicon(self) -> Dict[str, Dict[str, List[str]]]:
        return self._lemma_lexicon

    def _remove_abbr_first_words_that_are_lex_words(self) -> None:
        """We don't want to tag 'loc.' in e.g. 'au adus-o pe loc.' as an abbreviation."""

        for word in self._abbrfirstword1:
            if not self.is_lex_word(word) or \
                    self.is_msd_rxonly_word(word, Lex._abbr_pos_pattern):
                self._abbrfirstword.add(word)
            # end if
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

        def _add_to_infl_class(iclass: Dict[str, Dict[str, List[str]]], lemma: str, word: str, msd: str):
            if lemma not in iclass:
                iclass[lemma] = {}
            # end if

            if msd not in iclass[lemma]:
                iclass[lemma][msd] = []
            # end if

            if word not in iclass[lemma][msd]:
                iclass[lemma][msd].append(word)
            # end if
        # end def

        with open(TBL_WORDFORM_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                counter += 1

                if counter % 100000 == 0:
                    logger.info(
                        f'Read [{counter}] lines from file [{TBL_WORDFORM_FILE}]')
                # end if

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

                    # Record all inflected forms for a lemma
                    if Lex._simple_word_pattern.match(lemma) and \
                            len(lemma) > 1 and \
                            (msd.startswith('Nc') or msd.startswith('Afp') or msd.startswith('Vm')):
                        if msd.startswith('Nc'):
                            infl_class = self._inflectional_class['noun']
                        elif msd.startswith('Afpm'):
                            infl_class = self._inflectional_class['adje_masc']
                        elif msd.startswith('Afpf'):
                            infl_class = self._inflectional_class['adje_fem']
                        elif msd.startswith('Vm'):
                            infl_class = self._inflectional_class['verb']
                        # end if

                        if Lex._verb_no_number_pattern.fullmatch(msd):
                            msd_s = msd + 's'
                            msd_p = msd + 'p'

                            _add_to_infl_class(infl_class, lemma, word, msd_s)
                            _add_to_infl_class(infl_class, lemma, word, msd_p)
                        # end if

                        if msd.startswith('Afp-'):
                            for infl_class in [
                                self._inflectional_class['adje_masc'],
                                self._inflectional_class['adje_fem']
                            ]:
                                _add_to_infl_class(
                                    infl_class, lemma, word, msd)
                            # end for
                        else:
                            _add_to_infl_class(infl_class, lemma, word, msd)
                        # end if
                    # end if

                    diac_word = word
                    no_diac_word = self._get_romanian_word_with_no_diacs(word)

                    self._add_word_to_lex(diac_word, msd, lemma)
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

    def _aclass_noun_adj_msd(self, word: str, msd: str) -> str:
        """If MSD is adjective, return the noun equivalent.
        And the other way around."""

        msd2 = ''

        if msd.startswith('Afp'):
            msd3 = msd.replace('Afp', 'Nc')

            if (self.is_lex_word(word) and self.can_be_msd(word, msd3)) or \
                    not self.is_lex_word(word):
                msd2 = msd3
            # end if
        elif msd.startswith('Nc'):
            msd3 = msd.replace('Nc', 'Afp')

            if (self.is_lex_word(word) and self.can_be_msd(word, msd3)) or \
                    not self.is_lex_word(word):
                msd2 = msd3
            # end if
        # end if

        if msd2 and self._msd.is_valid_msd(msd2):
            return msd2
        else:
            return ''
        # end if

    def _aclass_prop_noun_msd(self, word: str, msd: str) -> str:
        msd2 = ''

        if Lex.sentence_case_pattern.match(word) and msd.startswith('Nc'):
            msd2 = msd.replace('Nc', 'Np')
        # end if

        if msd2 and self._msd.is_valid_msd(msd2):
            return msd2
        else:
            return ''
        # end if

    def _aclass_adj_part_msd(self, word: str, msd: str) -> str:
        def _check_and_return(m1: str, m2: str) -> str:
            if msd == m1 and self.can_be_msd(word, m2):
                return m2
            elif msd == m2 and self.can_be_msd(word, m1):
                return m1
            else:
                return ''
            # end if
        # end def

        for m1, m2 in [('Afpfsrn', 'Vmp--sf'), ('Afpfp-n', 'Vmp--pf'),
                       ('Afpms-n', 'Vmp--sm'), ('Afpmp-n', 'Vmp--pm')]:
            msd2 = _check_and_return(m1, m2)

            if msd2:
                return msd2
            # end if

            msd2 = _check_and_return(m2, m1)

            if msd2:
                return msd2
            # end if
        # end for

        return ''

    def amend_ambiguity_class(self, word: str, aclass: Set[str]) -> Set[str]:
        """Checks for common MSD patterns and completes the ambiguity class."""

        result_aclass = set()

        for m in aclass:
            result_aclass.add(m)

            for m2 in [
                    self._aclass_adj_part_msd(word, m),
                    self._aclass_noun_adj_msd(word, m),
                    self._aclass_prop_noun_msd(word, m)]:
                if m2:
                    result_aclass.add(m2)
                    log_once(
                        f"Added MSD '{m2}' to word '{word}'",
                        calling_fn='Lex.amend_ambiguity_class()',
                        log_level=logging.DEBUG)
                # end if
            # end for
        # end for

        return result_aclass

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

    def _add_word_to_lex(self, word: str, msd: str, lemma: str = ''):
        if not self._msd.is_valid_msd(msd):
            raise RuntimeError(f'MSD {msd} is not valid for word {word}')
        # end if

        if word not in self._lexicon:
            self._lexicon[word] = set()
        # end if

        self._lexicon[word].add(msd)

        if lemma:
            if word not in self._lemma_lexicon:
                self._lemma_lexicon[word] = {}
            # end if

            if msd not in self._lemma_lexicon[word]:
                self._lemma_lexicon[word][msd] = []
            # end if

            if lemma not in self._lemma_lexicon[word][msd]:
                self._lemma_lexicon[word][msd].append(lemma)
            # end if
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

    def get_word_ambiguity_class(self, word: str, exact_match: bool = False) -> List[str]:
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

    def get_unknown_ambiguity_class(self, word: str) -> List[str]:
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

        log_once(
            f"Unknown word [{word}] has ambiguity class [{', '.join([x for x in word_amb_class])}]",
            calling_fn='Lex.get_unknown_ambiguity_class()',
            log_level=logging.DEBUG)

        return word_amb_class

    def _get_possible_msds_from_prefix(self, word: str, min_pref_len: int = 2) -> Set[str]:
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

        # 3. Concatenate 1, 1.1, and 2
        return np.concatenate((features1, features11, features2))

    def get_word_lemma(self, word: str, msd: str) -> list:
        if word in self._lemma_lexicon and \
                msd in self._lemma_lexicon[word]:
            return self._lemma_lexicon[word][msd]
        # end if

        exact_match = False

        # For some of the word forms such as 'NaCl2' or 'Radu/Np',
        # we require an exact match in the lexicon.
        if Lex.mixed_case_pattern.match(word) or \
                Lex.code_case_pattern.match(word) or \
                msd.startswith('Np') or \
                msd.startswith('Yn'):
            exact_match = True
        # end if

        if not exact_match:
            word = word.lower()

            if word in self._lemma_lexicon and \
                    msd in self._lemma_lexicon[word]:
                return self._lemma_lexicon[word][msd]
            # end if
        # end if

        return []
