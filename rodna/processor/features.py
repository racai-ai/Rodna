from typing import List, Tuple
import numpy as np
import unicodedata as uc
from .lexicon import Lex
from .. import logger


class RoFeatures(object):
    """This class will generate Romanian-specific features for POS tagging."""

    # Special POS tagging features for Romanian
    # Computed here for any Romanian sentence
    # Format: str: int (which is the feature index)
    romanian_pos_tagging_features = {
        # Word is at the beginning of the sentence
        "WORD_AT_BOS": 0,
        # Word is at the end of the sentence
        "WORD_AT_EOS": 1,
        # THE ABOVE TWO FEATURES ARE NOT TO BE DELETED!
        # Word is some form of 'a fi'
        "TO_BE_AUX": 2,
        # Resolve Afp/Rgp ambiguity
        "SHOULD_BE_ADVERB": 3,
        "SHOULD_BE_ADJECTIVE": 4,
        "SHOULD_BE_NOUN": 5,
        "SHOULD_BE_VINF": 6
    }

    def __init__(self, lexicon: Lex) -> None:
        self._lexicon = lexicon

    def compute_sentence_wide_features(self, sentence: List[Tuple[str, str]]) -> None:
        """Will take a sentence and update it with sentence-wide POS tagging features."""

        for i in range(len(sentence)):
            parts = sentence[i]
            sentence[i] = (parts[0], parts[1], [])
        # end for

        self._set_bos_and_eos(sentence)
        self._set_to_be_aux(sentence)
        self._set_adjective(sentence)
        self._set_adverb(sentence)
        self._set_verb_inf(sentence)

    def get_context_feature_vector(self, computed_features: list) -> np.ndarray:
        features = np.zeros(
            len(RoFeatures.romanian_pos_tagging_features), dtype=np.float32)

        for f in computed_features:
            features[RoFeatures.romanian_pos_tagging_features[f]] = 1.0
        # end for

        return features

    def _set_bos_and_eos(self, sentence: List[Tuple[str, str, List[str]]]):
        sentence[0] = (sentence[0][0], sentence[0][1], ["WORD_AT_BOS"])
        sentence[-1] = (sentence[-1][0], sentence[-1][1], ["WORD_AT_EOS"])

    def _set_adjective(self, sentence: List[Tuple[str, str, List[str]]]):
        """Patterns:
        - cel/cea/cei/cele/celui/celei/celor mai ADJECTIV"""

        for i in range(len(sentence)):
            word = sentence[i][0]

            if word.lower() in ['cel', 'cea', 'cei', 'cele', 'celui', 'celei', 'celor'] and \
                    i < len(sentence) - 2:
                next_word = sentence[i + 1][0]
                next_next_word = sentence[i + 2][0]

                if next_word.lower() == 'mai' and self._lexicon.can_be_msd(next_next_word, "Afp"):
                    sentence[i + 2][2].append("SHOULD_BE_ADJECTIVE")
                # end if
            # end if
        # end for

    def _set_adverb(self, sentence: List[Tuple[str, str, List[str]]]):
        """Patterns:
        - VERB ADVERB
        - A FI + ADVERB + ADJECTIV"""

        for i in range(len(sentence)):
            word = sentence[i][0]

            if i < len(sentence) - 2:
                next_word = sentence[i + 1][0]
                next_next_word = sentence[i + 2][0]

                if self._lexicon.is_to_be_word(word) and self._lexicon.can_be_msd(next_word, "Rgp") and \
                    self._lexicon.can_be_msd(next_next_word, "Afp"):
                    sentence[i + 1][2].append("SHOULD_BE_ADVERB")
                # end if
            # end if

            if i > 1:
                prev_word = sentence[i - 1][0]

                if self._lexicon.can_be_msd(prev_word, "Vm") and self._lexicon.can_be_msd(word, "Rgp"):
                    sentence[i][2].append("SHOULD_BE_ADVERB")
                # end if
        # end for

    def _set_verb_inf(self, sentence: List[Tuple[str, str, List[str]]]):
        """Patterns:
        - a ... <VINF>
        - putea <VINF>"""

        for i in range(len(sentence)):
            word = sentence[i][0]

            if i > 2 and self._lexicon.can_be_msd(word, "Vmnp"):
                prev_word = sentence[i - 1][0]
                prev_prev_word = sentence[i - 2][0]

                if prev_word.lower() == 'a' or prev_prev_word.lower() == 'a':
                    sentence[i][2].append("SHOULD_BE_VINF")
                # end if
            # end if

            if i < len(sentence) - 1 and self._lexicon.is_can_word(word):
                next_word = sentence[i + 1][0]

                if self._lexicon.can_be_msd(next_word, "Vmnp"):
                    sentence[i + 1][2].append("SHOULD_BE_VINF")
                # end if
            # end if
        # end for

    def _set_to_be_aux(self, sentence: List[Tuple[str, str, List[str]]]):
        """Patterns:
        - a fi ... ADJECTIV/PARTICIPIU
        - a fi ... NOUN (fara prepozitie)"""

        for i in range(len(sentence)):
            word_i = sentence[i][0]

            if self._lexicon.is_to_be_word(word_i):
                for j in range(i + 1, len(sentence)):
                    word_j = sentence[j][0]

                    if self._lexicon.can_be_msd(word_j, "R") or \
                            self._lexicon.can_be_msd(word_j, "T") or \
                            self._lexicon.can_be_msd(word_j, "S"):
                        continue
                    elif self._lexicon.has_ambiguity_class(word_j, "Afp", "Vmp"):
                        sentence[i][2].append("TO_BE_AUX")
                        break
                    # end if
                # end for
            # end if
        # end for

        for i in range(len(sentence)):
            word_i = sentence[i][0]

            if self._lexicon.is_to_be_word(word_i):
                for j in range(i + 1, len(sentence)):
                    word_j = sentence[j][0]

                    if self._lexicon.can_be_msd(word_j, "R") or \
                            self._lexicon.can_be_msd(word_j, "T") or \
                            (self._lexicon.can_be_msd(word_j, "Afp") and not self._lexicon.can_be_msd(word_j, "Nc")):
                        continue
                    elif self._lexicon.can_be_msd(word_j, "Nc") and "TO_BE_AUX" not in sentence[i][2]:
                        sentence[i][2].append("TO_BE_AUX")
                        break
                    # end if
                # end for
            # end if
        # end for


class CharUni(object):
    """Creates a dictionary of Unicode properties from given input strings.
    Saves and loads the property statistics to/from the given file."""

    def __init__(self):
        self._seenunicodeid = 1
        self._seenunicodeprops = {"_UNK": 0}

    def add_unicode_props(self, input_string: str) -> None:
        for c in input_string:
            try:
                c_name = uc.name(c)
                name_words = c_name.split()
            except ValueError:
                if c == '\r' or c == '\n':
                    name_words = ['EOL']
                elif c == '\t':
                    name_words = ['TAB']
                else:
                    name_words = []
                # end if
            # end try

            for c_cat in name_words:
                if c_cat not in self._seenunicodeprops:
                    self._seenunicodeprops[c_cat] = self._seenunicodeid
                    self._seenunicodeid += 1
                # end if
            # end for
        # end for

    def save_unicode_props(self, file: str):
        logger.info(f"Saving file [{file}]")

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
        logger.info(f"Loading file [{file}]")

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
                if c == '\r' or c == '\n':
                    name_words = ['EOL']
                elif c == '\t':
                    name_words = ['TAB']
                else:
                    name_words = []
                # end if
            # end try

            for c_cat in name_words:
                if c_cat in self._seenunicodeprops:
                    result[self._seenunicodeprops[c_cat]] = 1.0
                    found = True
                # end if
            # end for
        # end for

        # If word has no known properties, set the unknown one.
        if not found:
            result[0] = 1.0
        # end if

        return result
