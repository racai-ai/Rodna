import numpy as np
import re
from config import MSD_MAP_FILE

class MSD(object):
    """This class will model a Morpho-Syntactic Descriptor.
    It will return a real-valued, fixed-length vector for a MSD."""

    punct_ctag = 'PUNCT'
    # Both MSD and CTAG have this
    unknown_label = 'X'
    unknown_punct_msd = 'Z'
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
        ("PuncType", ("c", "p", "l", "s", "d", "q", "x", "h", "o", "1", "2")),
        ("PuncPairType", ("p", "q")),
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
        "Z": ("PuncType", "PuncPairType")
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

    @staticmethod
    def get_msd_pos(msd: str) -> str:
        if msd:
            return msd[0]
        else:
            return '?'
        # end if

    def get_input_vector_size(self):
        return self._msdinputsize

    def get_output_vector_size(self):
        return self._msdoutputsize

    def get_ctag_output_vector_size(self):
        return self._ctagoutputsize

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

    def get_x_input_vector(self) -> np.ndarray:
        """Returns the input vector for the MSD 'X'."""

        msd_vec = np.zeros(self._msdinputsize, dtype=np.float32)
        xi = self._msd_poses.index(MSD.unknown_label)
        msd_vec[xi] = 1.0

        return msd_vec

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
