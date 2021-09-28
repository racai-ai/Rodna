import numpy as np
import re
from config import MSD_MAP_FILE

class MSD(object):
    """This class will model a Morpho-Syntactic Descriptor.
    It will return a real-valued, fixed-length vector for a MSD."""

    _punct_msd_list = [
        # Any other punctuation
        "Z",
        # Comma
        "Zc",
        # Period
        "Zp",
        # Colon
        "Zl",
        # Semicolon
        "Zs",
        # Dash
        "Zd",
        # Question mark
        "Zq",
        # Exclamation mark
        "Zx",
        # Ellipsis
        "Zh",
        # Pair tags
        # Z1 = open punctuation
        # Z2 = close punctuation
        # Parenthesis
        "Z1p",
        "Z2p",
        # Single/double quotes
        "Z1q",
        "Z2q"
    ]
    _punct_msd_inventory = {
        ".": "Zp",
        ",": "Zc",
        ":": "Zl",
        ";": "Zs",
        "-": "Zd",
        "?": "Zq",
        "!": "Zx",
        "...": "Zh",
        "…": "Zh",
        "(": "Z1p",
        "[": "Z1p",
        "{": "Z1p",
        ")": "Z2p",
        "]": "Z2p",
        "}": "Z2p",
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
    _punct_patt = re.compile("^\\W+$")

    def __init__(self):
        """Takes the MSD -> CTAG map file, commonly named msdtag.ro.map"""
        # This is the list of lists of possible MSD attributes
        self._vectorattrs = []
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
        self._msdinputsize = -1
        self._create_msd_vector_attrs(MSD_MAP_FILE)

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

    def idx_to_msd(self, index: int) -> str:
        if index < 0 or index >= len(self._msdinverseinv):
            return "Unk"
        else:
            return self._msdinverseinv[index]
        # end if

    def msd_to_idx(self, msd: str) -> int:
        if msd in self._msdinventory:
            return self._msdinventory[msd]
        else:
            return -1
        # end if

    def is_valid_msd(self, msd: str) -> bool:
        return msd in self._msdinventory

    def _create_msd_vector_attrs(self, mapfile: str):
        """Constructs the vector of possible MSD attributes."""

        vector_sets = []

        # 1. Read all MSDs and put their attributes in positional sets.
        with open(mapfile, mode="r") as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                msd = parts[0]

                # Store MSD in inventory, so that we know
                # which are the valid MSDs
                if msd not in self._msdinventory:
                    self._msdinverseinv.append("")
                    self._msdinventory[msd] = self._msdoutputsize
                    self._msdinverseinv[self._msdoutputsize] = msd

                    if msd == "X":
                        self._xindex = self._msdoutputsize
                    # end if

                    self._msdoutputsize += 1
                # end if

                # Store each MSD positional attribute in its own set
                for i in range(len(msd)):
                    if i == len(vector_sets):
                        vector_sets.append(set())
                    # end if

                    vector_sets[i].add(msd[i])
                # end for i
            # end for all lines
        # end with

        # 1.1 Add punctuation MSDs as well
        for msd in MSD._punct_msd_list:
            if msd not in self._msdinventory:
                self._msdinverseinv.append("")
                self._msdinventory[msd] = self._msdoutputsize
                self._msdinverseinv[self._msdoutputsize] = msd
                self._msdoutputsize += 1
            # end if

            for i in range(len(msd)):
                if i == len(vector_sets):
                    vector_sets.add(set())
                # end if

                vector_sets[i].add(msd[i])
            # end for
        # end for

        # 2. Sort each set and add it to the vector attributes
        first_position = True
        vec_size = 0

        for attrs in vector_sets:
            attrs_list = list(attrs)
            attrs_list.sort()

            if first_position:
                for i in range(len(attrs_list)):
                    if attrs_list[i] == "X":
                        self._xattrindex = i
                    # end if
                # end for i
            # end if

            self._vectorattrs.append(attrs_list)
            vec_size += len(attrs_list)
            first_position = False
        # end for attrs

        # 3. Set the vector size.
        self._msdinputsize = vec_size

    def get_x_input_vector(self) -> np.ndarray:
        """Returns the input vector for the MSD 'X'."""

        msd_vec = np.zeros(self._msdinputsize, dtype=np.float32)
        msd_vec[self._xattrindex] = 1.0

        return msd_vec

    def get_x_reference_vector(self):
        return self.msd_reference_vector("X")

    def msd_input_vector(self, msd: str) -> np.ndarray:
        """This method takes a str MSD and returns the numpy
        binary vector representation for it."""

        # If MSD is not in inventory, return the X representation
        if msd not in self._msdinventory:
            return self.get_x_input_vector()
        # end if

        msd_vec = np.zeros(self._msdinputsize, dtype=np.float32)
        vector_offset = 0

        for i in range(len(msd)):
            attr_i = msd[i]
            attrs_i = self._vectorattrs[i]

            for j in range(len(attrs_i)):
                if attrs_i[j] == attr_i:
                    msd_vec[vector_offset + j] = 1.0
                    break
                # end if
            # end for j

            vector_offset += len(attrs_i)
        # end for i

        return msd_vec

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

