import sys
from inspect import stack
import unicodedata as uc
import numpy as np


class CharUni(object):
    """Creates a dictionary of Unicode properties from given input strings.
    Saves and loads the property statistics to/from the given file."""

    def __init__(self):
        self._seenunicodeid = 1
        self._seenunicodeprops = {"_UNK": 0}

    def add_unicode_props(self, input_string: str) -> None:
        for c in input_string:
            c_name = uc.name(c)
            name_words = c_name.split()

            for c_cat in name_words:
                if c_cat not in self._seenunicodeprops:
                    self._seenunicodeprops[c_cat] = self._seenunicodeid
                    self._seenunicodeid += 1
                # end if
            # end for
        # end for

    def save_unicode_props(self, file: str):
        print(stack()[0][3] + ": saving file {0}".format(
            file), file=sys.stderr, flush=True)

        with open(file, "w", encoding="utf-8") as f:
            # Write the next Unicode property ID
            print("{0}".format(self._seenunicodeid), file=f, flush=True)

            # Write the Unicode properties dictionary
            for prop in self._seenunicodeprops:
                print("{0}\t{1!s}".format(
                    prop, self._seenunicodeprops[prop]), file=f, flush=True)
            # end for
        # end with

    def load_unicode_props(self, file: str):
        print(stack()[0][3] + ": loading file {0}".format(
            file), file=sys.stderr, flush=True)

        first_line = True

        with open(file, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if first_line:
                    self._seenunicodeid = int(line)
                    first_line = False
                else:
                    parts = line.split()
                    self._seenunicodeprops[parts[0]] = int(parts[1])
                # end if
            # end for
        # end with

    def get_unicode_features(self, word: str) -> np.ndarray:
        result = np.zeros((len(self._seenunicodeprops)), dtype=np.float32)
        found = False

        for c in word:
            try:
                c_name = uc.name(c)
                name_words = c_name.split()
            except ValueError:
                name_words = []
            # end try

            for c_cat in name_words:
                if c_cat in self._seenunicodeprops:
                    result[self._seenunicodeprops[c_cat]] = 1.0
                    found = True
                # end if
            # end for

        # If word has no known properties, set the unknown one.
        if not found:
            result[0] = 1.0
        # end if

        return result
