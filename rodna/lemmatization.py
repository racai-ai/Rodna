from typing import List, Tuple
import os
import re
import sys
from math import log10
from random import shuffle, seed
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from .morphology import RoInflect
from utils.Lex import Lex
from config import PARADIGM_MORPHO_FILE, \
    TBL_WORDROOT_FILE, TBL_ROOT2ROOT_FILE, ROOT_EXTRACT_LOG_FILE

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
        'Mc-p-l',
        'Mc-s-d',
        'Mc-s-b',
        'Vmnp'
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

    _verb_inflect_class_msds = [
        # infinitive
        'Vmnp',
        # indicative, present
        'Vmip1s',
        'Vmip2s',
        'Vmip3s',
        'Vmip1p',
        'Vmip2p',
        'Vmip3p',
        # indicative, imperfect
        'Vmii1s',
        'Vmii2s',
        'Vmii3s',
        'Vmii1p',
        'Vmii2p',
        'Vmii3p',
        # indicative, past
        'Vmis1s',
        'Vmis2s',
        'Vmis3s',
        'Vmis1p',
        'Vmis2p',
        'Vmis3p',
        # indicative, pluperfect
        'Vmil1s',
        'Vmil2s',
        'Vmil3s',
        'Vmil1p',
        'Vmil2p',
        'Vmil3p',
        # imperative
        'Vmm-2s',
        'Vmm-2p',
        # gerund
        'Vmg',
        # participle
        'Vmp--pm',
        'Vmp--sm',
        'Vmp--sf',
        'Vmp--pf',
        # conjunctive
        'Vmsp3'
    ]

    def __init__(self, lexicon: Lex, inflector: RoInflect):
        # This is the object recognizing the ambiguity class of a given word.
        self._roinflect = inflector
        self._lexicon = lexicon
        paras, term2paras = self._read_paradigm_morph()
        self._paradigms = paras
        self._terms_to_paradigms = term2paras

        if os.path.exists(TBL_WORDROOT_FILE):
            self._word_roots = self._read_root_lexicon()
        else:
            self._word_roots = self._build_root_lexicon()
        # end if

        if os.path.exists(TBL_ROOT2ROOT_FILE):
            self._root_rules = self._read_root_to_lemma_rules()
        else:
            self._root_rules = self._build_root_to_lemma_rules()
        # end if

    # Done
    def _s2s_rule(self, left: str, right: str) -> tuple:
        """String to string changing rule. Change `left` string
        into the `right` string."""

        if left == right:
            return tuple(['nop'])
        # endif

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

    # Done
    def _build_roots(self, infl_class: dict, infl_msds: list, para_type: str) -> list:
        """Builds the nouns, adjectives and verbs root lexicon from tbl.wordform.ro and morphalt.xml."""

        found_roots = []
        best_match_count = 0

        for pname in self._paradigms:
            if ((para_type.startswith('noun') or para_type.startswith('adje')) and \
                pname.startswith('verb')) or \
                (para_type == 'verb' and not pname.startswith('verb')):
                continue
            # end if

            alternative_roots = {}

            # 0. Populate alternative roots in this paradigm
            for msd in self._paradigms[pname]:
                for ending, alter in self._paradigms[pname][msd]:
                    if alter not in alternative_roots:
                        alternative_roots[alter] = ''
                    # end if
                # end for
            # end for

            # 1. Check if this paradigm is of type being requested for.
            # E.g a nominal feminine inflectional class against a nomfem paradigm.
            pname_not_matching = False

            for msd in infl_msds:
                # Paradigm is not suitable for the set of MSDs being matched.
                if msd not in self._paradigms[pname]:
                    pname_not_matching = True
                    break
                # end if
            # end for

            if pname_not_matching:
                continue
            # end if

            # match_count + not_found_count == len(infl_msds)
            match_count = 0
            not_matched_msds = []

            # 2. Get the wordform for the MSD of the inflectional class
            # and match its ending with the ending for the MSD of the
            # current paradigm.
            for msd in infl_msds:
                wordforms = []

                if (para_type == 'noun' or para_type == 'verb') and \
                        msd in infl_class:
                    wordforms = infl_class[msd]
                elif para_type.startswith('adje'):
                    amsd = msd.replace('Nc', 'Afp')

                    if amsd in infl_class:
                        wordforms = infl_class[amsd]
                    # end if
                # end if

                # Ignore missing wordforms for the inflectional MSD set.
                if not wordforms:
                    continue
                # end if

                wordform_matched = False

                for wordform in wordforms:
                    for ending, alter in self._paradigms[pname][msd]:
                        if wordform.endswith(ending):
                            wf_root = re.sub(ending + '$', '', wordform)

                            if not alternative_roots[alter]:
                                alternative_roots[alter] = wf_root
                                wordform_matched = True
                                match_count += 1
                            elif alternative_roots[alter] == wf_root:
                                wordform_matched = True
                                match_count += 1
                            # end if

                            if wordform_matched:
                                break
                            # end if
                        # end if

                        if wordform_matched:
                            break
                        # end if
                    # end for
                # end for

                if not wordform_matched:
                    not_matched_msds.append(msd)
                # end if
            # end all MSD in inflectional class

            # 3. Check if all alternative roots
            # have been filled in.
            for alter in alternative_roots:
                if not alternative_roots[alter]:
                    pname_not_matching = True
                    break
                # end if
            # end all alternate roots

            if pname_not_matching:
                continue
            # end if

            # 4. Score the paradigm
            if match_count > best_match_count:
                best_match_count = match_count
                found_roots = [(alternative_roots, pname, not_matched_msds)]
            elif match_count == best_match_count:
                found_roots.append((alternative_roots, pname, not_matched_msds))
            # end if
        # end all paradigms

        return found_roots

    # Done
    def _read_root_lexicon(self) -> dict:
        roots = {}

        print(f'Reading word root lexicon from [{TBL_WORDROOT_FILE}]', file=sys.stderr, flush=True)

        with open(TBL_WORDROOT_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                r = line.strip().split()
                lemma = r[0]
                pos = r[1]
                paradigm = r[-1]
                alt_roots = {}

                for i in range(2, len(r) - 1):
                    alt, root = r[i].split('/')
                    alt_roots[alt] = root
                # end for

                if pos not in roots:
                    roots[pos] = {}
                # end if

                if lemma not in roots[pos]:
                    roots[pos][lemma] = []
                # end if

                roots[pos][lemma].append((alt_roots, paradigm))
            # end for
        # end with

        return roots

    # Done
    def _build_root_lexicon(self) -> dict:
        infl_classes = self._lexicon.get_inflectional_classes()
        
        with open(ROOT_EXTRACT_LOG_FILE, mode='w', encoding='utf-8') as ff:
            with open(TBL_WORDROOT_FILE, mode='w', encoding='utf-8') as f:
                for pos in infl_classes:
                    for lemma in tqdm(sorted(infl_classes[pos]), desc=f'Roots[{pos}]: '):
                        result = []
                        infl_class = infl_classes[pos][lemma]

                        if ('Ncms-n' in infl_class and 'Ncfp-n' in infl_class) or \
                                ('Afpms-n' in infl_class and 'Afpfp-n' in infl_class):
                            # Neuter noun
                            result = self._build_roots(infl_class,
                                                RoLemmatizer._nom_neu_inflect_class_msds, para_type=pos)
                        elif ('Ncfsrn' in infl_class and 'Ncfp-n' in infl_class) or \
                                ('Afpfsrn' in infl_class and 'Afpfp-n' in infl_class):
                            # Feminine noun                    
                            result = self._build_roots(infl_class,
                                                RoLemmatizer._nom_fem_inflect_class_msds, para_type=pos)
                        elif ('Ncms-n' in infl_class and 'Ncmp-n' in infl_class) or \
                                ('Afpms-n' in infl_class and 'Afpmp-n' in infl_class):
                            # Masculine noun
                            result = self._build_roots(infl_class,
                                                RoLemmatizer._nom_masc_inflect_class_msds, para_type=pos)
                        # end if

                        if 'Vmnp' in infl_class:
                            # Verb
                            result = self._build_roots(infl_class,
                                                RoLemmatizer._verb_inflect_class_msds, para_type=pos)
                        # end if

                        for alt_roots, para_name, nm_msds in result:
                            if nm_msds:
                                for nmm in nm_msds:
                                    if pos.startswith('adje'):
                                        nmm = nmm.replace('Nc', 'Afp')
                                    # end if
                                    
                                    print(
                                        f'Could not match [{infl_class[nmm]}/{nmm}] in paradigm {para_name}', file=ff)
                            else:
                                print(f'{lemma}\t{pos}\t', file=f, end='')

                                for alt_no in sorted(list(alt_roots.keys())):
                                    alt_root = alt_roots[alt_no]
                                    print(f'{alt_no}/{alt_root}\t', file=f, end='')
                                # end for

                                print(f'{para_name}', file=f)
                            # end if
                        # end for
                    # end for all lemmas
                # end noun, adje, verb
            # end with
        # end with

        return self._read_root_lexicon()

    # Done
    def _read_root_to_lemma_rules(self) -> dict:
        rules = {}

        print(
            f'Reading lemmatization rules from [{TBL_ROOT2ROOT_FILE}]', file=sys.stderr, flush=True)

        with open(TBL_ROOT2ROOT_FILE, mode='r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                paradigm = parts[0]
                alt = parts[1]
                rule = tuple(['' if x == '--' else x for x in parts[2:-1]])
                freq = int(parts[-1])

                if paradigm not in rules:
                    rules[paradigm] = {}
                # end if

                if alt not in rules[paradigm]:
                    rules[paradigm][alt] = {}
                # end if

                rules[paradigm][alt][rule] = freq
            # end for
        # end with

        return rules

    # Done
    def _build_root_to_lemma_rules(self) -> dict:
        """Learns a set of root changes into lemmas."""
       
        rules = {}

        with open(TBL_ROOT2ROOT_FILE, mode='w', encoding='utf-8') as f:
            for pos in self._word_roots:
                for lemma in tqdm(sorted(self._word_roots[pos]), desc=f'Rules[{pos}]: '):
                    for alternatives, paradigm in self._word_roots[pos][lemma]:
                        for alt in alternatives:
                            root = alternatives[alt]
                            rule = self._s2s_rule(root, lemma)

                            if paradigm not in rules:
                                rules[paradigm] = {}
                            # end if

                            if alt not in rules[paradigm]:
                                rules[paradigm][alt] = {}
                            # end if

                            if rule not in rules[paradigm][alt]:
                                rules[paradigm][alt][rule] = 1
                            else:
                                rules[paradigm][alt][rule] += 1
                            # end if
                    # end all paradigms for lemma
                # end all lemmas
            # end all poses

            # Write rules to file
            for paradigm in rules:
                for alt in rules[paradigm]:
                    for rule in rules[paradigm][alt]:
                        rule2 = [x if x else '--' for x in rule]
                        rule_string = '\t'.join(rule2)
                        freq = rules[paradigm][alt][rule]
                        print('\t'.join([paradigm, alt, rule_string, str(freq)]), file=f)
                    # end for
                # end for
            # end for
        # end with

        return rules

    # Done
    def _parse_attr_values(self, attr_value: str) -> list:
        if attr_value.startswith('{') and attr_value.endswith('}'):
            attr_values = attr_value[1:-1].split()
        else:
            attr_values = [attr_value]
        # end if

        return attr_values

    # Done
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

    # Done
    def _parse_nomsuff_paradigm(self, paradigm: Element) -> dict:
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

                                            if not alt_val:
                                                alt_val = '1'
                                            # end if

                                            endings.append((ending_val, alt_val))
                                        # end for
                                    # end if
                                # end for
                            else:
                                for term_elem in case_elem.findall('TERM'):
                                    ending_val = term_elem.get('TERM')
                                    alt_val = term_elem.get('ALT')

                                    if not alt_val:
                                        alt_val = '1'
                                    # end if

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

    # Done
    def _parse_nom_paradigm(self, paradigm: Element, gender: str, nomsuff: bool = False) -> dict:
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

                                        if not alt_val:
                                            alt_val = '1'
                                        # end if

                                        endings.append((ending_val, alt_val))
                                    # end for
                                # end if
                            # end for
                        else:
                            for term_elem in case_elem.findall('TERM'):
                                ending_val = term_elem.get('TERM')
                                alt_val = term_elem.get('ALT')

                                if not alt_val:
                                    alt_val = '1'
                                # end if

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

    # Done
    def _edit_verb_endings(self, msd_to_endings: dict):
        if msd_to_endings['Vmsp1s'] == msd_to_endings['Vmip1s']:
            del msd_to_endings['Vmsp1s']
        # end if

        if msd_to_endings['Vmsp2s'] == msd_to_endings['Vmip2s']:
            del msd_to_endings['Vmsp2s']
        # end if

        if msd_to_endings['Vmsp1p'] == msd_to_endings['Vmip1p']:
            del msd_to_endings['Vmsp1p']
        # end if

        if msd_to_endings['Vmsp2p'] == msd_to_endings['Vmip2p']:
            del msd_to_endings['Vmsp2p']
        # end if

        if msd_to_endings['Vmsp3s'] == msd_to_endings['Vmsp3p']:
            msd_to_endings['Vmsp3'] = msd_to_endings['Vmsp3s']
            del msd_to_endings['Vmsp3s']
            del msd_to_endings['Vmsp3p']
        # end if

        msd_to_endings['Vmg-------y'] = set()
        vmg_end, vmg_alt = list(msd_to_endings['Vmg'])[0]
        msd_to_endings['Vmg-------y'].add((vmg_end + 'u', vmg_alt))
        msd_to_endings['Vmp--sm---y'] = set()
        vmp_end, vmp_alt = list(msd_to_endings['Vmp--sm'])[0]
        msd_to_endings['Vmg-------y'].add((vmp_end + 'u', vmp_alt))

    # Done
    def _parse_verb_paradigm(self, paradigm: Element) -> dict:
        """Parses a verbal paradigm. Returns the MSD to endigs dictionary."""
        voice_elem = paradigm.find('VOICE')
        msd_to_endings = {}
        current_msd = ['Vm']

        def _set_m2e(msd: str, above_elem: Element):
            endings = []

            for term_elem in above_elem.findall('TERM'):
                ending = term_elem.get('TERM')
                alternative = term_elem.get('ALT')

                if not alternative:
                    alternative = '1'
                # end if

                endings.append((ending, alternative))
            # end for term

            if msd not in msd_to_endings:
                msd_to_endings[msd] = set()
            # end if

            msd_to_endings[msd].update(endings)
        # end def

        for tensed_elem in voice_elem.findall('TENSED'):
            mood = tensed_elem.get('MOOD')

            if mood == 'infinitive':
                # Special case when infinitive is on the
                # tensed element...
                _set_m2e('Vmnp', above_elem=tensed_elem)
                continue
            # end if

            for mood_elem in tensed_elem.findall('MOOD'):
                mood = mood_elem.get('MOOD')

                if mood == 'indicative':
                    current_msd.append('i')

                    for tense_elem in mood_elem.findall('TENSE'):
                        tense = tense_elem.get('TENSE')

                        if tense == 'present':
                            current_msd.append('p')
                        elif tense == 'imperfect':
                            current_msd.append('i')
                        elif tense == 'simpleperfect':
                            current_msd.append('s')
                        elif tense == 'pastperfect':
                            current_msd.append('l')
                        # end if

                        for num_elem in tense_elem.findall('NUM'):
                            number = num_elem.get('NUM')

                            for pers_elem in num_elem.findall('PERS'):
                                person = pers_elem.get('PERS')

                                if number == 'singular':
                                    current_msd.append(person + 's')
                                else:
                                    current_msd.append(person + 'p')
                                # end if

                                msd = ''.join(current_msd)
                                _set_m2e(msd, above_elem=pers_elem)
                                current_msd.pop()
                            # end for person
                        # end for number

                        current_msd.pop()
                    # end for tense

                    current_msd.pop()
                # end for mood
                elif mood == 'conjunctive':
                    current_msd.append('sp')

                    for num_elem in mood_elem.findall('NUM'):
                        number = num_elem.get('NUM')

                        for pers_elem in num_elem.findall('PERS'):
                            person = pers_elem.get('PERS')

                            if number == 'singular':
                                current_msd.append(person + 's')
                            else:
                                current_msd.append(person + 'p')
                            # end if

                            msd = ''.join(current_msd)
                            _set_m2e(msd, above_elem=pers_elem)
                            current_msd.pop()
                        # end for person
                    # end for number

                    current_msd.pop()
                elif mood == 'imperative':
                    current_msd.append('m')

                    for num_elem in mood_elem.findall('NUM'):
                        number = num_elem.get('NUM')

                        if number == 'singular':
                            for neg_elem in num_elem.findall('NEG'):
                                negation = neg_elem.get('NEG')

                                if negation == 'yes':
                                    continue
                                # end if

                                current_msd.append('-2s')
                                msd = ''.join(current_msd)
                                _set_m2e(msd, above_elem=neg_elem)
                                current_msd.pop()
                            # end for neg
                        elif number == 'plural':
                            current_msd.append('-2p')
                            msd = ''.join(current_msd)
                            _set_m2e(msd, above_elem=num_elem)
                            current_msd.pop()
                        # end if
                    # end for number

                    current_msd.pop()
                elif mood == 'infinitive':
                    current_msd.append('np')
                    msd = ''.join(current_msd)
                    _set_m2e(msd, above_elem=mood_elem)
                    current_msd.pop()
                elif mood == 'participle':
                    current_msd.append('p')

                    for voice_elem in mood_elem.findall('VOICE'):
                        voice = voice_elem.get('VOICE')

                        if voice == 'passive':
                            for gender_elem in voice_elem.findall('GEN'):
                                gender = gender_elem.get('GEN')

                                for num_elem in gender_elem.findall('NUM'):
                                    number = num_elem.get('NUM')

                                    if number == 'singular':
                                        if gender == 'masculine':
                                            current_msd.append('--sm')
                                        else:
                                            current_msd.append('--sf')
                                        # end if
                                    else:
                                        if gender == 'masculine':
                                            current_msd.append('--pm')
                                        else:
                                            current_msd.append('--pf')
                                        # end if
                                    # end if

                                    msd = ''.join(current_msd)
                                    _set_m2e(msd, above_elem=num_elem)
                                    current_msd.pop()
                                # end for num
                            # end for gender
                        # end if
                    # end for voice

                    current_msd.pop()
                elif mood == 'gerund':
                    current_msd.append('g')
                    msd = ''.join(current_msd)
                    _set_m2e(msd, above_elem=mood_elem)
                    current_msd.pop()
                # end if
            # end for mood
        # end for tensed

        self._edit_verb_endings(msd_to_endings)
        return msd_to_endings

    # Done
    def _read_paradigm_morph(self) -> tuple:
        parser = ET.XMLParser()
        document = ET.parse(PARADIGM_MORPHO_FILE, parser)
        root = document.getroot()
        paradigms = {}
        term_to_paradigms = {}
        
        for paradigm in root:
            pname = paradigm.get('PARADIGM')

            if pname.startswith('nomverbsuf') or \
                    pname.startswith('verbsufpart') or \
                    pname.startswith('verbsufger'):
                continue
            # end if

            if pname.startswith('nomsuf'):
                print(f'Parsing paradigm [{pname}]',
                      file=sys.stderr, flush=True)
                gender = paradigm.get('GEN')
                m2e = self._parse_nom_paradigm(paradigm, gender, nomsuff=True)
                paradigms[pname] = m2e
            elif pname.startswith('nom') or pname.startswith('voc'):
                print(f'Parsing paradigm [{pname}]',
                      file=sys.stderr, flush=True)
                gender = paradigm.get('GEN')
                m2e = self._parse_nom_paradigm(paradigm, gender)
                paradigms[pname] = m2e
            elif pname.startswith('verb'):
                print(f'Parsing paradigm [{pname}]',
                      file=sys.stderr, flush=True)
                m2e = self._parse_verb_paradigm(paradigm)
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

    def _transform_root_to_lemma(self, root: str, paradigm: str, altno: str) -> list:
        """This method takes the root of the word, after the inflexional
        ending has been removed and transforms it into the lemma, given paradigm and
        alternative number. For instance `canioanele` has root `canioan` -> `canion`."""

        candidates = []

        if paradigm in self._root_rules and \
                altno in self._root_rules[paradigm]:
            for rule in self._root_rules[paradigm][altno]:
                lemma = ''
                r_action = rule[0]
                r_what = ''

                if r_action != 'nop':
                    r_what = rule[1]
                # end if

                freq = self._root_rules[paradigm][altno][rule]

                if r_action == 'add':
                    lemma = root + r_what
                elif r_action == 'chg' and root.endswith(r_what):
                    r_with = rule[2]
                    lemma = re.sub(r_what + '$', r_with, root)
                elif r_action == 'del' and root.endswith(r_what):
                    lemma = re.sub(r_what + '$', '', root)
                elif r_action == 'nop':
                    lemma = root
                # end if
                
                if lemma:
                    candidates.append((lemma, freq))
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

    def _find_paradigm_msd(self, msd: str, paradigm) -> str:
        """If MSD in underspecified, search for a specified
        MSD in the paradigm."""

        if msd in self._paradigms[paradigm]:
            return msd
        # end if

        if msd.startswith('Nc'):
            if 'verb' in paradigm:
                return ''
            # end if

            if msd == 'Nc':
                if 'Ncms-n' in self._paradigms[paradigm]:
                    return 'Ncms-n'
                elif 'Ncfsrn' in self._paradigms[paradigm]:
                    return 'Ncfsrn'
                # end if
            else:
                for msd2 in self._paradigms[paradigm]:
                    if (msd[2] == msd2[2] or msd[2] == '-') and \
                        (msd[3] == msd2[3] or msd[3] == '-') and \
                        (msd[4] == msd2[4] or msd[4] == '-'):
                        return msd2
                    # end if
                # end for
            # end if
        elif msd.startswith('Vm'):
            if 'verb' not in paradigm:
                return ''
            # end if

            if (msd.endswith('1') or msd.endswith('2') or msd.endswith('3')):
                msd2 = msd + 's'

                if msd2 in self._paradigms[paradigm]:
                    return msd2
                # end if

                msd2 = msd + 'p'

                if msd2 in self._paradigms[paradigm]:
                    return msd2
                # end if
            # end if
        # end if

        return ''

    def _correct_wordform(self, word: str, msd: str) -> tuple:
        """Some wordforms are incorrect. Attempt to fix them here
        so that we can lemmatize them."""

        if msd.startswith('Ncm') and msd.endswith('yy') and word.endswith('u'):
            # e.g. 'curentu', 'băiatu', etc.
            word += 'l'
            msd = msd[0:-1]
            print(f'Correct word [{word}] and MSD [{msd}]', file=sys.stderr, flush=True)
            return (word, msd)
        # end if

        if msd == 'Vmil2p' and word.endswith('seți'):
            word = re.sub('seți$', 'serăți', word)
            print(f'Correct word [{word}] and MSD [{msd}]', file=sys.stderr, flush=True)
            return (word, msd)
        # end if

        if msd == 'Vmil1p' and word.endswith('sem'):
            word = re.sub('sem$', 'serăm', word)
            print(f'Correct word [{word}] and MSD [{msd}]', file=sys.stderr, flush=True)
            return (word, msd)
        # end if

        if msd.startswith('Vm') and msd.endswith('-y') and word.endswith('u'):
            msd = re.sub('-+y$', '', msd)
            word = word[0:-1]
            print(f'Correct word [{word}] and MSD [{msd}]', file=sys.stderr, flush=True)
            return (word, msd)
        # end if

        return (word, msd)

    # Done
    def _generate_possible_paradigms(self, word: str, msd: str) -> list:
        """Word is already lower cased."""
        paradigm_candidates = []
        empty_ending = ''

        # Include the empty ending in the search
        if empty_ending in self._terms_to_paradigms:
            paradigm_candidates_empty = []

            for pname in self._terms_to_paradigms[empty_ending]:
                msd2 = self._find_paradigm_msd(msd, pname)
                
                if msd2:
                    for pterm, alt in self._paradigms[pname][msd2]:
                        if word.endswith(pterm):
                            # Found a possible paradigm for the word
                            paradigm_candidates_empty.append(
                                (pname, pterm, alt))
                        # end if
                    # end for
                # end if MSD is in paradigm
            # end all paradigms for term

            if paradigm_candidates_empty:
                paradigm_candidates.extend(paradigm_candidates_empty)
            # end if
        # end if term is in paradigms index

        for i in range(1, len(word)):
            term = word[-i:]

            if term in self._terms_to_paradigms:
                paradigm_candidates_term = []

                for pname in self._terms_to_paradigms[term]:
                    msd2 = self._find_paradigm_msd(msd, pname)
                    
                    if msd2:
                        for pterm, alt in self._paradigms[pname][msd2]:
                            if word.endswith(pterm):
                                # Found a possible paradigm for the word
                                paradigm_candidates_term.append(
                                    (pname, pterm, alt))
                            # end if
                        # end for
                    # end if MSD is in paradigm
                # end all paradigms for term

                if paradigm_candidates_term:
                    paradigm_candidates.extend(paradigm_candidates_term)
                # end if
            # end if term is in paradigms index
        # end all terminations

        return paradigm_candidates

    def _get_lemma_msd(self, paradigm: str, msd: str) -> str:
        if msd.startswith('Afp'):
            return 'Afpms-n'
        elif 'Ncfsrn' in self._paradigms[paradigm]:
            return 'Ncfsrn'
        elif 'Ncms-n' in self._paradigms[paradigm]:
            return 'Ncms-n'
        elif 'Vmnp' in self._paradigms[paradigm]:
            return 'Vmnp'
        # end if

        return msd

    def lemmatize_sentence(self, sentence: List[Tuple]) -> List[Tuple]:
        """Takes a sentence from the RoPOSTagger, a list of (wordform, MSD, score), and
        produces a list of (wordform, MSD, lemma, score)."""
        
        result = []

        for word, msd, _ in sentence:
            lemmas = self.lemmatize(word, msd)
            lemma_found = False

            for lemma, score, m in lemmas:
                if m == msd:
                    result.append((word, msd, lemma, score))
                    lemma_found = True
                    break
                # end if
            # end for

            if not lemma_found:
                # If no lemma found, just put the word...
                result.append((word, msd, word, 0.01))
            # end if
        # end for

        return result

    def lemmatize(self, word: str, msd: str, use_lex: bool = True) -> list:
        """Main lemmatization method. If `use_lex` is `True`, if word/MSD is in the
        lexicon, return the looked-up lemma. If not, try and find the most probable paradigm
        based on the supplied word form and MSD and get the lemma from there."""

        # 1. Do lexicon-based lemmatization
        if use_lex:
            lex_lemmas = self._lexicon.get_word_lemma(word, msd)

            if lex_lemmas:
                return [(x, 1.0, msd) for x in lex_lemmas]
            # end if
        # end if

        # 2. If MSD is of a word that is already in lemma form,
        # just return the word
        if msd in RoLemmatizer._lemma_msds:
            if msd.startswith('Np') or msd.startswith('Y') or \
                    msd.startswith('Z'):
                return [(word, 1.0, msd)]
            else:
                return [(word.lower(), 1.0, msd)]
            # end if
        # end if

        # 3. If MSD is that of a punctuation mark,
        # just return the punctuation
        if msd.startswith('Z'):
            return [(word, 1.0, msd)]
        # end if

        lcword = word.lower()
        lcword, msd = self._correct_wordform(lcword, msd)
        paradigm_candidates = []
        nmsd = msd

        # 3. Do the lemmatization of the unknown word
        if msd.startswith('Nc') or msd.startswith('Afp'):
            if msd.startswith('Afp'):
                nmsd = msd.replace('Afp', 'Nc')
            # end if

            paradigm_candidates = self._generate_possible_paradigms(
                lcword, nmsd)
        elif msd.startswith('Vm'):
            paradigm_candidates = self._generate_possible_paradigms(
                lcword, msd)
        # end if

        lemma_candidates = []

        # 4. Main lemmatization algorithm.
        # - remove the ending
        # - transform root to lemma based on rules
        for pname, pterm, altno in paradigm_candidates:
            word_root = re.sub(pterm + '$', '', lcword)
            lmsd = self._get_lemma_msd(pname, msd)

            for lem, frq in self._transform_root_to_lemma(word_root, pname, altno):
                lemma_candidates.append((lem, frq, lmsd, pname))
            # end for
        # end for

        lemma_scores = {}

        # 5. Compute lemma scores based on how many paradigms generated it
        # Also use our NN morphology module to say how likely the lemma is
        for l, f, m, p in lemma_candidates:
            if (l, m) not in lemma_scores:
                sc = self._roinflect.msd_prob_for_word(l, m)
                lemma_scores[(l, m)] = {'SCORE': sc, 'FREQ': f, 'PARADIGMS': [p]}
            else:
                lemma_scores[(l, m)]['FREQ'] += f
                lemma_scores[(l, m)]['PARADIGMS'].append(p)
            # end if
        # end for

        possible_lemma_msds = {}
        lemma_scores2 = []

        for lemma, msd in lemma_scores:
            nn = lemma_scores[(lemma, msd)]['SCORE']
            fq = \
                lemma_scores[(lemma, msd)]['FREQ'] / \
                len(lemma_scores[(l, m)]['PARADIGMS'])
            sc = (nn * nn) * log10(fq)
            lemma_scores2.append((lemma, sc, msd))

            if msd not in possible_lemma_msds:
                possible_lemma_msds[msd] = nn
            else:
                possible_lemma_msds[msd] += nn
            # end if
        # end for

        lemma_scores3 = []

        for l, s, m in lemma_scores2:
            lemma_scores3.append((l, s * possible_lemma_msds[m], m))
        # end for

        final_lemmas = sorted(lemma_scores3, key=lambda x: x[1], reverse=True)

        if not final_lemmas:
            print(f'Empty response from lemmatizer for {word}/{msd}', file=sys.stderr, flush=True)
        # end if

        return final_lemmas

    def test(self, sample_size: int = 1000):
        """Executes a test test with wordforms from the lexicon,
        without using the lexicon for lemmatization.
        Computes accuracy and MRR for the returned lemmas."""

        # Get same results from the random number generator
        seed(1234)

        infl_classes = self._lexicon.get_inflectional_classes()
        samples_per_pos = int(sample_size / len(infl_classes))
        test_set = []
        word_rx = re.compile('^[a-zșțăîâ]+$')

        for pos in infl_classes:
            pos_lemmas = list(infl_classes[pos].keys())
            shuffle(pos_lemmas)

            for i in range(min(samples_per_pos, len(pos_lemmas))):
                l = pos_lemmas[i]
                lemma_msds = list(infl_classes[pos][l])
                shuffle(lemma_msds)
                m = lemma_msds.pop(0)
                w = infl_classes[pos][l][m][0]

                while lemma_msds and (not word_rx.fullmatch(w) or \
                    m == 'Ncfsrn' or m == 'Ncms-n' or \
                    m == 'Afpms-n' or m == 'Vmnp'):
                    m = lemma_msds.pop(0)
                    w = infl_classes[pos][l][m][0]
                # end while

                if m != 'Ncfsrn' and m != 'Ncms-n' and \
                    m != 'Afpms-n' and m != 'Vmnp' and \
                    word_rx.fullmatch(w):
                    test_set.append((w, m, l))
                # end if
            # end for
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
        print(f'Accuracy@1: {accuracy1:.5f}')
        print(f'Accuracy@2: {accuracy2:.5f}')
        print(f'MRR: {mrr:.5f}')
        print(f'Recall: {recall:.5f}')


if __name__ == '__main__':
    lexi = Lex()
    morpho = RoInflect(lexi)
    morpho.load()
    lemmi = RoLemmatizer(lexi, morpho)
    lemmi.test()

    #while True:
    #    print("> ", end='')
    #    word, msd = sys.stdin.readline().strip().split()
    #    lemmas = lemmi.lemmatize(word, msd, use_lex=False)
    #    print(lemmas)
    # end while
