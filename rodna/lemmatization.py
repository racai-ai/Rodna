import os
import re
import sys
from random import shuffle, seed
from tqdm import tqdm
from lxml import etree
from lxml.etree import _Element
from .morphology import RoInflect
from utils.Lex import Lex
from config import PARADIGM_MORPHO_FILE, \
    TBL_WORDROOT_FILE, TBL_ROOT2ROOT_FILE

class RoLemmatizer(object):
    """This is the Romanian lemmatizer, using the Romanian Paradigmatic Morphology."""

    _fix_msd_map = {
        'Npmsrn': 'Npms-n',
        'Ncmsrn': 'Ncms-n',
        'Npmson': 'Npms-n',
        'Ncmson': 'Ncms-n',
        'Npmprn': 'Npmp-n',
        'Ncmprn': 'Ncmp-n',
        'Npmpon': 'Npmp-n',
        'Ncmpon': 'Ncmp-n',
        'Npmpvn': 'Npmp-n',
        'Ncmpvn': 'Ncmp-n',
        'Ncfprn': 'Ncfp-n',
        'Npfprn': 'Npfp-n',
        'Ncfpon': 'Ncfp-n',
        'Npfpon': 'Npfp-n',
        'Ncfpvn': 'Ncfp-n',
        'Npfpvn': 'Npfp-n'
    }

    _lemma_msds = {
        'Af',
        'Afp',
        'Afpm--n',
        'Ncms-n',
        'Afpms-n',
        'Ncfsrn',
        'Npfsrn',
        'Afpfsrn',
        'Ncm--n',
        'Nc',
        'Ncf--n',
        'Nc---n',
        'Ncmsrn',
        'Npms-n',
        'Np',
        'Npfsry',
        'Npfpry',
        'Npmsry',
        'Y',
        'Yn',
        'Rgp',
        'I',
        'Mc-p-l'
    }

    # For the following 3, nom. sg. and nom. pl. MSDs are at 0 and 1!
    _nom_masc_inflect_class_msds = [
        # băiat
        'Ncms-n',
        # băieț-i
        'Ncmp-n',
        # băiat-ul
        'Ncmsry',
        # băiat-ului
        'Ncmsoy',
        # băiat-ule
        'Ncmsvy',
        # băieț-ii
        'Ncmpry',
        # băieț-ilor
        'Ncmpoy',
        # băieț-ilor
        'Ncmpvy'
    ]

    _nom_fem_inflect_class_msds = [
        # fat-ă
        'Ncfsrn',
        # fet-e
        'Ncfp-n',
        # fat-a
        'Ncfsry',
        # fet-e,
        'Ncfson',
        # fet-ei
        'Ncfsoy',
        # fat-o
        'Ncfsvy',
        # fet-ele
        'Ncfpry',
        # fet-elor
        'Ncfpoy',
        # fet-elor
        'Ncfpvy'
    ]

    _nom_neu_inflect_class_msds = [
        # canion
        'Ncms-n',
        # canioan-e
        'Ncfp-n',
        # canion-ul
        'Ncmsry',
        # canion-ului
        'Ncmsoy',
        # canion-ule
        'Ncmsvy',
        # canioan-ele
        'Ncfpry',
        # canioan-elor
        'Ncfpoy',
        # canioan-elor
        'Ncfpvy'
    ]

    def __init__(self, lexicon: Lex, inflector: RoInflect):
        # This is the object recognizing the ambiguity class of a given word.
        self._roinflect = inflector
        self._lexicon = lexicon
        paras, term2paras = self._read_paradigm_morph()
        self._paradigms = paras
        self._terms_to_paradigms = term2paras
        self._nn_weight = 0.5
        self._fq_weight = 0.5

        if os.path.exists(TBL_WORDROOT_FILE):
            self._word_roots = self._read_root_lexicon()
        else:
            self._word_roots = self._build_root_lexicon()
        # end if

        if os.path.exists(TBL_ROOT2ROOT_FILE):
            self._root_rules = self._read_root_changing_rules()
        else:
            self._root_rules = self._build_root_changing_rules()
        # end if

    def _s2s_rule(self, left: str, right: str) -> tuple:
        """String to string changing rule. Change `left` string
        into the `right` string."""

        if len(left) <= len(right):
            i = 0

            while i < len(left) and left[i] == right[i]:
                i += 1
            # end while

            if i == len(left):
                return ('add', right[i:])
            else:
                return ('chg', left[i:], right[i:])
            # end if
        else:
            i = 0

            while i < len(right) and left[i] == right[i]:
                i += 1
            # end while

            if i == len(right):
                return ('del', left[i:])
            else:
                return ('chg', left[i:], right[i:])
            # end if
        # end if

    def _build_nom_root_rules(self) -> dict:
        """Learns the string transformation rules from
        a plural nominal root of a noun/adjective to its singular root."""

        nominal_rules = {}

        for lemma in tqdm(self._word_roots, desc='Nominal rules: '):
            for pos, sg_root, pl_root, paradigm, _ in self._word_roots[lemma]:
                if (pos == 'noun' or pos == 'adje'):
                    if sg_root != pl_root:
                        ps_rule = self._s2s_rule(pl_root, sg_root)
                        sp_rule = self._s2s_rule(sg_root, pl_root)
                    else:
                        ps_rule = ('none', '', '')
                        sp_rule = ('none', '', '')
                    # end if

                    if paradigm not in nominal_rules:
                        nominal_rules[paradigm] = {'sg-pl': {}, 'pl-sg': {}}
                    # end if

                    if ps_rule not in nominal_rules[paradigm]['pl-sg']:
                        nominal_rules[paradigm]['pl-sg'][ps_rule] = 1
                    else:
                        nominal_rules[paradigm]['pl-sg'][ps_rule] += 1
                    # end if

                    if sp_rule not in nominal_rules[paradigm]['sg-pl']:
                        nominal_rules[paradigm]['sg-pl'][sp_rule] = 1
                    else:
                        nominal_rules[paradigm]['sg-pl'][sp_rule] += 1
                    # end if
                # end if nominal
            # end all roots
        # end all lemmas

        return nominal_rules

    def _build_nom_roots(self, lemma: str, infl_classes: dict, infl_msds: list) -> list:
        """Builds the nouns and adjectives root lexicon from tbl.wordform.ro and morphalt.xml."""

        best_paradigms = []
        best_nominal_pairs = []
        max_count = 0
        can_be = '--'
        what_is = ''
        sg_msd = infl_msds[0]
        pl_msd = infl_msds[1]

        for pname in self._paradigms:
            pncount = 0
            pacount = 0
            nominals = {sg_msd: [], pl_msd: []}

            for nmsd in infl_msds:
                if nmsd in self._paradigms[pname]:
                    amsd = nmsd.replace('Nc', 'Afp')

                    if nmsd in infl_classes[lemma]:
                        best_word = ''
                        best_term = ''
                        
                        for word in infl_classes[lemma][nmsd]:
                            for term, _ in self._paradigms[pname][nmsd]:
                                if word.endswith(term):
                                    if not best_word or len(term) > len(best_term):
                                        best_word = word
                                        best_term = term
                                    # end if
                                # end if
                            # end all endings
                        # end all words for a lemma/msd

                        if best_word:
                            pncount += 1

                            if nmsd == sg_msd or nmsd == pl_msd:
                                nominals[nmsd].append(best_word)
                            # end if
                        # end if
                    elif amsd in infl_classes[lemma]:
                        best_word = ''
                        best_term = ''

                        for word in infl_classes[lemma][amsd]:
                            for term, _ in self._paradigms[pname][nmsd]:
                                if word.endswith(term):
                                    if not best_word or len(term) > len(best_term):
                                        best_word = word
                                        best_term = term
                                    # end if
                                # end if
                            # end all endings
                        # end all words for a lemma/msd

                        if best_word:
                            pacount += 1

                            if nmsd == sg_msd or nmsd == pl_msd:
                                nominals[nmsd].append(best_word)
                            # end if
                        # end if
                    # end if
                # end if
            # end all nominal MSDs

            if nominals[sg_msd] and nominals[pl_msd]:
                if pncount >= pacount:
                    if pncount > max_count:
                        max_count = pncount
                        best_paradigms = [pname]
                        best_nominal_pairs = [nominals]
                        what_is = 'noun'

                        if pacount > 0:
                            can_be = 'adje'
                        # end if
                    elif pncount == max_count:
                        best_paradigms.append(pname)
                        best_nominal_pairs.append(nominals)
                    # end if
                else:
                    if pacount > max_count:
                        max_count = pacount
                        best_paradigms = [pname]
                        best_nominal_pairs = [nominals]
                        what_is = 'adje'

                        if pncount > 0:
                            can_be = 'noun'
                        # end if
                    elif pacount == max_count:
                        best_paradigms.append(pname)
                        best_nominal_pairs.append(nominals)
                    # end if
                # end if
            # end if
        # end all paradigms

        result = []

        for i in range(len(best_paradigms)):
            best_paradigm = best_paradigms[i]
            best_nominals = best_nominal_pairs[i]

            for sg_wordform in best_nominals[sg_msd]:
                sg_root = ''
                
                for term, _ in self._paradigms[best_paradigm][sg_msd]:
                    if sg_wordform.endswith(term):
                        sg_root = re.sub(term + '$', '', sg_wordform)
                        break
                    # end if
                # end for

                for pl_wordform in best_nominals[pl_msd]:
                    pl_root = ''

                    for term, _ in self._paradigms[best_paradigm][pl_msd]:
                        if pl_wordform.endswith(term):
                            pl_root = re.sub(term + '$', '', pl_wordform)
                            break
                        # end if
                    # end for

                    if sg_root and pl_root:
                        # Avoid adding empty roots!
                        result.append((what_is, sg_root, pl_root, best_paradigm, can_be))
                    # end if
                # end for pl
            # end for sg
        # end for

        result_equal_roots = []
        result_equal_lenghts_roots = []

        # If we have the same root for the sg. and pl. forms,
        # choose that paradigm.
        for r in result:
            if r[1] == r[2]:
                result_equal_roots.append(r)
            elif len(r[1]) == len(r[2]):
                result_equal_lenghts_roots.append(r)
            # end fi
        # end for

        if result_equal_roots:
            return result_equal_roots
        elif result_equal_lenghts_roots:
            return result_equal_lenghts_roots
        else:
            return result
        # end if

    def _read_root_lexicon(self) -> dict:
        roots = {}

        print(f'Reading word root lexicon from [{TBL_WORDROOT_FILE}]', file=sys.stderr, flush=True)

        with open(TBL_WORDROOT_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                r = line.strip().split()
                lemma = r[0]
                r = tuple(r[1:])

                if lemma not in roots:
                    roots[lemma] = [r]
                else:
                    roots[lemma].append(r)
                # end if
            # end for
        # end with

        return roots

    def _check_infl_class(self, lemma: str, infl_classes: dict) -> tuple:
        n = 0
        a = 0
        v = 0

        for msd in infl_classes[lemma]:
            if msd.startswith('Nc'):
                n += 1
            elif msd.startswith('Afp'):
                a += 1
            elif msd.startswith('Vm'):
                v += 1
            # end if
        # end for

        return (n, a, v)

    def _build_root_lexicon(self) -> dict:
        infl_classes = self._lexicon.get_inflectional_classes()
        roots = {}
        
        with open(TBL_WORDROOT_FILE, mode='w', encoding='utf-8') as f:
            for lemma in tqdm(sorted(infl_classes), desc='Roots: '):
                nc, ac, _ = self._check_infl_class(lemma, infl_classes)

                # Check for nominal interpretation
                if nc == 0 and ac == 0:
                    continue
                # end if

                result = []

                if ('Ncms-n' in infl_classes[lemma] and 'Ncfp-n' in infl_classes[lemma]) or \
                        ('Afpms-n' in infl_classes[lemma] and 'Afpfp-n' in infl_classes[lemma]):
                    # Neuter noun
                    result = self._build_nom_roots(lemma, infl_classes,
                                        RoLemmatizer._nom_neu_inflect_class_msds)

                    if not result:
                        print(
                            f'Error: could not find any neuter paradigm for lemma [{lemma}]',
                            file=sys.stderr, flush=True)
                    # end if
                elif ('Ncfsrn' in infl_classes[lemma] and 'Ncfp-n' in infl_classes[lemma]) or \
                        ('Afpfsrn' in infl_classes[lemma] and 'Afpfp-n' in infl_classes[lemma]):
                    # Feminine noun                    
                    result = self._build_nom_roots(lemma, infl_classes,
                                        RoLemmatizer._nom_fem_inflect_class_msds)

                    if not result:
                        print(
                            f'Error: could not find any feminine paradigm for lemma [{lemma}]',
                            file=sys.stderr, flush=True)
                    # end if
                elif ('Ncms-n' in infl_classes[lemma] and 'Ncmp-n' in infl_classes[lemma]) or \
                        ('Afpms-n' in infl_classes[lemma] and 'Afpmp-n' in infl_classes[lemma]):
                    # Masculine noun
                    result = self._build_nom_roots(lemma, infl_classes,
                                        RoLemmatizer._nom_masc_inflect_class_msds)

                    if not result:
                        print(
                            f'Error: could not find any masculine paradigm for lemma [{lemma}]',
                            file=sys.stderr, flush=True)
                    # end if
                # end if

                for r in result:
                    print(lemma + '\t' + '\t'.join(r), file=f)
                # end for

                if result:
                    roots[lemma] = result
                # end if
            # end for all lemmas
        # end with

        return roots

    def _read_root_changing_rules(self) -> dict:
        rules = {}

        print(
            f'Reading lemmatization rules from [{TBL_ROOT2ROOT_FILE}]', file=sys.stderr, flush=True)

        with open(TBL_ROOT2ROOT_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                paradigm = parts[0]
                direction = parts[1]
                rule = tuple(['' if x == '--' else x for x in parts[2:-1]])
                freq = int(parts[-1])

                if paradigm.startswith('nom'):
                    if paradigm not in rules:
                        rules[paradigm] = {direction: {}}
                    elif direction not in rules[paradigm]:
                        rules[paradigm][direction] = {}
                    # end if

                    rules[paradigm][direction][rule] = freq
                # end if
            # end for
        # end with

        return rules

    def _build_root_changing_rules(self) -> dict:
        """Learns a set of pl. root to sg. root changes and
        the reverse of it."""
       
        rules = {}

        with open(TBL_ROOT2ROOT_FILE, mode='w', encoding='utf-8') as f:
            nom_rules = self._build_nom_root_rules()

            for paradigm in nom_rules:
                for direction in nom_rules[paradigm]:
                    for rule in nom_rules[paradigm][direction]:
                        rule2 = [x if x else '--' for x in rule]
                        rule_string = '\t'.join(rule2)
                        freq = nom_rules[paradigm][direction][rule]
                        print('\t'.join([paradigm, direction, rule_string, str(freq)]), file=f)
                    # end for
            # end for

            rules.update(nom_rules)
        # end with

        return rules

    def _parse_attr_values(self, attr_value: str) -> list:
        if attr_value.startswith('{') and attr_value.endswith('}'):
            attr_values = attr_value[1:-1].split()
        else:
            attr_values = [attr_value]
        # end if

        return attr_values

    def _map_noun_msd_to_endings(self,
            type_values: list, gender: str,
            num_values: list, case_values: list, encl_values: list,
            msd_to_endings: dict, endings: list) -> str:
        """Generates the MSDs from the attribute values and maps them to the specified endings."""

        if not gender:
            gv = '-'
        elif gender == 'masculine':
            gv = 'm'
        else:
            gv = 'f'
        # end if

        current_msd = ['N']

        # Here we generate the MSD with the ending info
        for t in type_values:
            if t == 'common':
                tv = 'c'
            else:
                tv = 'p'
            # end if

            current_msd.append(tv)
            current_msd.append(gv)

            for n in num_values:
                if n == 'singular':
                    nv = 's'
                else:
                    nv = 'p'
                # end if

                current_msd.append(nv)

                for c in case_values:
                    if c == 'nominative' or c == 'accusative':
                        cv = 'r'
                    elif c == 'genitive' or c == 'dative':
                        cv = 'o'
                    else:
                        cv = 'v'
                    # end if

                    current_msd.append(cv)

                    for cl in encl_values:
                        if cl == 'yes':
                            clv = 'y'
                        else:
                            clv = 'n'
                        # end if

                        current_msd.append(clv)
                        msd = ''.join(current_msd)

                        # Fix MSD so that we use an existing MSD
                        if msd in RoLemmatizer._fix_msd_map:
                            msd = RoLemmatizer._fix_msd_map[msd]
                        # end if

                        if msd not in msd_to_endings:
                            msd_to_endings[msd] = set()
                        # end if

                        msd_to_endings[msd].update(endings)
                        current_msd.pop()
                    # end for

                    current_msd.pop()
                # end for

                current_msd.pop()
            # end for

            current_msd.pop()
            current_msd.pop()
        # end for MSD

    def _parse_nomsuff_paradigm(self, paradigm: _Element) -> dict:
        msd_to_endings = {}

        for nt_elem in paradigm.findall('TYPE'):
            type_val = nt_elem.get('TYPE')
            type_values = self._parse_attr_values(type_val)

            for gen_elem in nt_elem.findall('GEN'):
                gender = gen_elem.get('GEN')

                for num_elem in gen_elem.findall('NUM'):
                    num_val = num_elem.get('NUM')
                    num_values = self._parse_attr_values(num_val)

                    for encl_elem in num_elem.findall('ENCL'):
                        encl_val = encl_elem.get('ENCL')
                        encl_values = self._parse_attr_values(encl_val)

                        for case_elem in encl_elem.findall('CASE'):
                            case_val = case_elem.get('CASE')
                            case_values = self._parse_attr_values(case_val)
                            endings = []

                            if case_val == 'vocative' or case_val == 'vocativ':
                                for hum_elem in case_elem.findall('HUM'):
                                    personal = hum_elem.get('HUM')

                                    if personal == 'person':
                                        for term_elem in hum_elem.findall('TERM'):
                                            ending_val = term_elem.get('TERM')
                                            alt_val = term_elem.get('ALT')
                                            endings.append((ending_val, alt_val))
                                        # end for
                                    # end if
                                # end for
                            else:
                                for term_elem in case_elem.findall('TERM'):
                                    ending_val = term_elem.get('TERM')
                                    alt_val = term_elem.get('ALT')
                                    endings.append((ending_val, alt_val))
                                # end for
                            # end if

                            self._map_noun_msd_to_endings(
                                type_values, gender, num_values,
                                case_values, encl_values, msd_to_endings, endings)
                        # end for CASE
                    # end for ENCL
                # end for NUM
            # end for GEN
        # end for

        return msd_to_endings

    def _parse_nom_paradigm(self, paradigm: _Element, gender: str, nomsuff: bool = False) -> dict:
        """If `nomsuff` is true, GEN is under TYPE.
        If `gender` is void, gender is an attribute of NUM."""

        if nomsuff:
            return self._parse_nomsuff_paradigm(paradigm)
        # end if

        msd_to_endings = {}

        for nt_elem in paradigm.findall('TYPE'):
            type_val = nt_elem.get('TYPE')
            type_values = self._parse_attr_values(type_val)

            for num_elem in nt_elem.findall('NUM'):
                num_val = num_elem.get('NUM')
                num_values = self._parse_attr_values(num_val)
                gen_value = num_elem.get('GEN')

                if not gen_value:
                    gen_value = gender
                # end if

                for encl_elem in num_elem.findall('ENCL'):
                    encl_val = encl_elem.get('ENCL')
                    encl_values = self._parse_attr_values(encl_val)

                    for case_elem in encl_elem.findall('CASE'):
                        case_val = case_elem.get('CASE')
                        case_values = self._parse_attr_values(case_val)
                        endings = []

                        if case_val == 'vocative' or case_val == 'vocativ':
                            for hum_elem in case_elem.findall('HUM'):
                                personal = hum_elem.get('HUM')

                                if personal == 'person':
                                    for term_elem in hum_elem.findall('TERM'):
                                        ending_val = term_elem.get('TERM')
                                        alt_val = term_elem.get('ALT')
                                        endings.append((ending_val, alt_val))
                                    # end for
                                # end if
                            # end for
                        else:
                            for term_elem in case_elem.findall('TERM'):
                                ending_val = term_elem.get('TERM')
                                alt_val = term_elem.get('ALT')
                                endings.append((ending_val, alt_val))
                            # end for
                        # end if

                        self._map_noun_msd_to_endings(
                            type_values, gen_value, num_values,
                            case_values, encl_values, msd_to_endings, endings)
                    # end for
                # end for
            # end for
        # end for

        return msd_to_endings

    def _read_paradigm_morph(self) -> tuple:
        parser = etree.XMLParser(resolve_entities=False)
        document = etree.parse(PARADIGM_MORPHO_FILE, parser)
        root = document.getroot()
        paradigms = {}
        term_to_paradigms = {}
        
        for paradigm in root:
            pname = paradigm.get('PARADIGM')
            gender = paradigm.get('GEN')
            print(f'Parsing paradigm [{pname}]', file=sys.stderr, flush=True)

            if pname.startswith('nomsuf'):
                m2e = self._parse_nom_paradigm(paradigm, gender, nomsuff=True)
                paradigms[pname] = m2e
            elif pname.startswith('nom') or pname.startswith('voc'):
                m2e = self._parse_nom_paradigm(paradigm, gender)
                paradigms[pname] = m2e
            # end if
        # end for

        for pname in paradigms:
            for msd in paradigms[pname]:
                for term, _ in paradigms[pname][msd]:
                    if term not in term_to_paradigms:
                        term_to_paradigms[term] = set()
                    # end if

                    term_to_paradigms[term].add(pname)
                # end all terminations
            # end all MSDs
        # end all paradigms

        return paradigms, term_to_paradigms

    def _transform_root(self, root: str, paradigm: str, direction: str) -> list:
        """This method takes the plural root of the word, after the inflexional
        ending has been removed and transforms it into the singular root in the paradigm.
        For instance `canioanele` has root `canioan` -> `canion`."""

        candidates = []

        if paradigm in self._root_rules:
            for rule in self._root_rules[paradigm][direction]:
                new_root = ''
                r_action = rule[0]
                r_what = rule[1]
                freq = self._root_rules[paradigm][direction][rule]

                if r_action == 'add':
                    new_root = root + r_what
                elif r_action == 'chg' and root.endswith(r_what):
                    r_with = rule[2]
                    new_root = re.sub(r_what + '$', r_with, root)
                elif r_action == 'del' and root.endswith(r_what):
                    new_root = re.sub(r_what + '$', '', root)
                elif r_action == 'none':
                    new_root = root
                # end if
                
                if new_root:
                    candidates.append((new_root, freq))
                # end if
            # end for

            if not candidates:
                candidates.append((root, 1))
            # end if    
        else:
            # No change
            candidates.append((root, 1))
        # end if paradigm

        return candidates

    def _get_nom_lemma_msd(self, nmsd: str, paradigm: str) -> str:
        if nmsd.startswith('Ncm'):
            return 'Ncms-n'
        elif nmsd.startswith('Ncf'):
            if paradigm.startswith('nomneu'):
                # For neuter gender nouns, get the masculine MSD
                return 'Ncms-n'
            else:
                return 'Ncfsrn'
            # end if
        # end if

    def _check_lemma_by_paradigm(self, lemma: str, lemm_msd: str, wordform: str, orig_msd: str, paradigm: str) -> bool:
        """Checks if obtained `lemma` with MSD `lemm_msd` can be inflected back to the
        original `wordform` with MSD `orig_msd`, using provided `paradigm`."""
        
        if lemm_msd in self._paradigms[paradigm] and \
                orig_msd in self._paradigms[paradigm]:
            for term, _ in self._paradigms[paradigm][lemm_msd]:
                if lemma.endswith(term):
                    root = re.sub(term + '$', '', lemma)

                    if orig_msd.startswith('Nc'):
                        if orig_msd[3] == 'p':
                            new_roots = self._transform_root(
                                root, paradigm, 'sg-pl')
                        else:
                            new_roots = [(root, 1)]
                        # end if
                    elif orig_msd.startswith('Vm'):
                        # TODO: add verbs here
                        new_roots = []
                    # end if

                    for term2, _ in self._paradigms[paradigm][orig_msd]:
                        for new_root, freq in new_roots:
                            new_wordform = new_root + term2

                            if wordform == new_wordform:
                                return True
                            # end if
                        # end for
                    # end for
                # end if
            # end for
        # end if

        return False
    
    def _produce_lemma(self, sg_root: str, paradigm: str, lemma_msd: str) -> list:
        """This methods takes the singular root of the word and produces its lemma,
        according to `paradigm` and the lemma MSD."""

        lemmas = []

        if paradigm in self._paradigms and \
                lemma_msd in self._paradigms[paradigm]:
            for term, _ in self._paradigms[paradigm][lemma_msd]:
                lemmas.append(sg_root + term)
            # end for
        else:
            lemmas.append(sg_root)
        # end if

        return lemmas

    def _generate_nom_possible_paradigms(self, word: str, nmsd: str) -> list:
        """Word is already lower cased."""
        paradigm_candidates = []

        for i in range(1, len(word)):
            term = word[-i:]

            if term in self._terms_to_paradigms:
                for pname in self._terms_to_paradigms[term]:
                    if nmsd in self._paradigms[pname]:
                        for pterm, _ in self._paradigms[pname][nmsd]:
                            if word.endswith(pterm):
                                # Found a possible paradigm for the word
                                paradigm_candidates.append((pname, pterm))
                            # end if
                        # end for
                    # end if MSD is in paradigm
                # end all paradigms for term
            # end if term is in paradigms index
        # end all terminations

        return paradigm_candidates

    def lemmatize(self, word: str, msd: str, use_lex: bool = True) -> list:
        """Main lemmatization method. If `use_lex` is `True`, if word/MSD is in the
        lexicon, return the looked-up lemma. If not, try and find the most probable paradigm
        based on the supplied word form and MSD and get the lemma from there."""

        # 1. Do lexicon-based lemmatization
        if use_lex:
            lex_lemmas = self._lexicon.get_word_lemma(word, msd)

            if lex_lemmas:
                return [(x, 1.0) for x in lex_lemmas]
            # end if
        # end if

        # 2. If MSD is of a word that is already in lemma form,
        # just return the word
        if msd in RoLemmatizer._lemma_msds:
            if msd.startswith('Np') or msd.startswith('Y'):
                return [(word, 1.0)]
            else:
                return [(word.lower(), 1.0)]
            # end if
        # end if

        lcword = word.lower()
        paradigm_candidates = []
        nmsd = msd

        # 3. Do the lemmatization of the unknown word
        if msd.startswith('Nc') or msd.startswith('Afp'):
            if msd.startswith('Afp'):
                nmsd = msd.replace('Afp', 'Nc')
            # end if

            paradigm_candidates = self._generate_nom_possible_paradigms(
                lcword, nmsd)
        elif msd.startswith('Vm'):
            # TODO: Lemmatize verbs
            pass
        # end if

        lemma_candidates = []

        # 4. Main lemmatization algorithm.
        # - remove the ending
        # - transform pl. root to sg. root based on rules
        # - append lemma ending to sg. root
        for pname, pterm in paradigm_candidates:
            word_root = re.sub(pterm + '$', '', lcword)

            if msd.startswith('Nc') or msd.startswith('Afp'):
                lmsd = self._get_nom_lemma_msd(nmsd, pname)
                possible_sg_roots = [(word_root, 1)]

                if nmsd[3] == 'p':
                    # Plural wordform, get singular roots
                    possible_sg_roots = self._transform_root(word_root, pname, 'pl-sg')
                # end if

                for sgr, frq in possible_sg_roots:
                    lemmas = self._produce_lemma(sg_root=sgr, paradigm=pname, lemma_msd=lmsd)

                    for lem in lemmas:
                        # 4.1 Validate final lemma by getting back the inflected wordform
                        # that produced it
                        if self._check_lemma_by_paradigm(lem, lmsd, lcword, nmsd, pname):
                            lemma_candidates.append((lem, frq, lmsd, pname))
                        # end if
                    # end for
                # end for
            elif msd.startswith('Vm'):
                # TODO: Lemmatize verbs
                pass
            # end if
        # end for

        lemma_scores = {}
        freq_sum = 0

        # 5. Compute lemma scores based on how many paradigms generated it
        # Also use our NN morphology module to say how likely the lemma is
        for l, f, m, p in lemma_candidates:
            sc = self._roinflect.msd_prob_for_word(l, m)

            if l not in lemma_scores:
                lemma_scores[l] = {'SCORE': sc, 'PARADIGMS': [(p, m)], 'RULEFREQ': f}
            else:
                lemma_scores[l]['SCORE'] += sc
                lemma_scores[l]['RULEFREQ'] += f
                lemma_scores[l]['PARADIGMS'].append((p, m))
            # end if

            freq_sum += f
        # end for

        lemma_scores2 = []       

        for lemma in lemma_scores:
            avg_prob = lemma_scores[lemma]['SCORE'] / \
                len(lemma_scores[lemma]['PARADIGMS'])
            freq = lemma_scores[lemma]['RULEFREQ']
            frq_prob = freq / freq_sum
            lemma_score = self._compute_lemma_score(
                nn_score=avg_prob, fq_score=frq_prob)
            lemma_scores2.append((lemma, lemma_score))
            freq_sum += freq
        # end for

        lemma_scores2 = sorted(lemma_scores2, key=lambda x: x[1], reverse=True)
        final_lemmas = []

        for lm, sc in lemma_scores2:
            if sc >= 0.001:
                final_lemmas.append((lm, sc))
            else:
                break
            # end if
        # end for

        if not final_lemmas:
            print(f'Empty response from lemmatizer for {word}/{msd}', file=sys.stderr, flush=True)
        # end if

        return final_lemmas

    def _compute_lemma_score(self, nn_score: float, fq_score: float) -> float:
        return self._nn_weight * nn_score + self._fq_weight * fq_score

    def set_lemma_score_weights(self, nnw: float, fqw: float):
        self._nn_weight = nnw
        self._fq_weight = fqw

    def test(self, sample_size: int = 1000):
        """Executes a test test with wordforms from the lexicon,
        without using the lexicon for lemmatization.
        Computes accuracy and MRR for the returned lemmas."""

        # Get same results from the random number generator
        seed(1234)

        lemma_lexicon = self._lexicon.get_lemma_lexicon()
        lexicon_words = list(lemma_lexicon.keys())
        shuffle(lexicon_words)
        noun_rx = re.compile('^Nc[mf][sp]')
        word_rx = re.compile('^[a-zșțăîâ]+$')
        test_set = []

        # For now, just for nouns, as adjectives have the masculine lemma.
        for w in lexicon_words:
            if word_rx.fullmatch(w):
                for m in lemma_lexicon[w]:
                    if noun_rx.match(m) and \
                            m != 'Ncfsrn' and m != 'Ncms-n':
                        for l in lemma_lexicon[w][m]:
                            test_set.append((w, m, l))
                        # end for
                    # end if

                    if len(test_set) == sample_size:
                        break
                    # end if
                # end for

                if len(test_set) == sample_size:
                    break
                # end if
            # end if
        # end for

        recall = 0
        mrr = 0
        accuracy1 = 0
        accuracy2 = 0

        # Test
        for w, m, l in tqdm(test_set, desc='Test: '):
            lemmas = self.lemmatize(word=w, msd=m, use_lex=False)

            if not lemmas:
                continue
            # end if

            if lemmas[0][0] == l:
                accuracy1 += 1
                accuracy2 += 1
            elif len(lemmas) > 1 and lemmas[1][0] == l:
                accuracy2 += 1
            # end if

            for i in range(len(lemmas)):
                pred_lemma = lemmas[i][0]

                if pred_lemma == l:
                    mrr += 1 / (i + 1)
                    recall += 1
                    break
                # end if
            # end for
        # end for

        accuracy1 /= len(test_set)
        accuracy2 /= len(test_set)
        recall /= len(test_set)
        mrr /= len(test_set)

        # Print results
        print(f'NN weight: {self._nn_weight}')
        print(f'FQ weight: {self._fq_weight}')
        print(f'Accuracy@1: {accuracy1:.5f}')
        print(f'Accuracy@2: {accuracy2:.5f}')
        print(f'MRR: {mrr:.5f}')
        print(f'Recall: {recall:.5f}')


if __name__ == '__main__':
    lexi = Lex()
    morpho = RoInflect(lexi)
    morpho.load()
    lemmi = RoLemmatizer(lexi, morpho)

    #for w in range(11):
    #    w = w / 10.
    #    lemmi.set_lemma_score_weights(w, 1 - w)
    #    lemmi.test()
    # end for

    while True:
        print("> ", end='')
        word, msd = sys.stdin.readline().strip().split()
        lemmas = lemmi.lemmatize(word, msd, use_lex=False)
        print(lemmas)
    # end while
