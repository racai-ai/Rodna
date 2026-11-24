import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict
from rodna.utils.datafile import read_conllu_file
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import unicodedata


def _align_sequences_dynamically(
    list_a: List[str],
    list_b: List[str],
    *,
    gap_penalty: float = 0.6,
    min_similarity: float = 0.9
) -> List[Tuple[int, int]]:
    """
    Find the best monotonic one-to-one alignment between two lists.

    Each element from `list_a` can be aligned with at most one element from
    `list_b`. Elements may also remain unaligned.

    Parameters
    ----------
    list_a, list_b:
        Ordered lists of elements.

    gap_penalty:
        Cost of leaving an element unaligned. A larger value favors more
        alignments; a smaller value permits more skipped elements.

    min_similarity:
        Minimum similarity required for two elements to be aligned.

    Returns
    -------
    List[Tuple[int, int]]
        Zero-based pairs (index_in_a, index_in_b), in textual order.
    """

    if gap_penalty < 0:
        raise ValueError("gap_penalty must be non-negative.")
    # end if

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between 0 and 1.")
    # end if

    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()

        return text
    # end def


    def similarity(text_a: str, text_b: str) -> float:
        # If text_a and text_b are only words
        if not text_a and not text_b:
            return 1.0
        elif not text_a or not text_b:
            return 0.0
        elif ' ' not in text_a and ' ' not in text_b:
            if text_a == text_b:
                return 1.0
            else:
                return 0.0
            # end if
        # end if

        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())

        return (
            2.0 * len(tokens_a & tokens_b)
            / (len(tokens_a) + len(tokens_b))
        )
    # end def

    normalized_a = [normalize(sentence) for sentence in list_a]
    normalized_b = [normalize(sentence) for sentence in list_b]

    n = len(list_a)
    m = len(list_b)
    infinity = float("inf")

    # dp[i][j] is the minimum cost of aligning list_a[:i] with list_b[:j].
    dp = [[infinity] * (m + 1) for _ in range(n + 1)]

    # Values are:
    #   "match": align A[i - 1] with B[j - 1]
    #   "skip_a": leave A[i - 1] unaligned
    #   "skip_b": leave B[j - 1] unaligned
    back: List[List[str | None]] = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
        back[i][0] = "skip_a"
    # end for

    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty
        back[0][j] = "skip_b"
    # end for

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Option 1: leave A[i - 1] unaligned.
            best_cost = dp[i - 1][j] + gap_penalty
            best_action = "skip_a"

            # Option 2: leave B[j - 1] unaligned.
            skip_b_cost = dp[i][j - 1] + gap_penalty

            if skip_b_cost < best_cost:
                best_cost = skip_b_cost
                best_action = "skip_b"
            # end if

            # Option 3: align A[i - 1] with B[j - 1].
            score = similarity(
                normalized_a[i - 1],
                normalized_b[j - 1],
            )

            if score >= min_similarity:
                match_cost = dp[i - 1][j - 1] + (1.0 - score)

                if match_cost < best_cost:
                    best_cost = match_cost
                    best_action = "match"
                # end if
            # end if

            dp[i][j] = best_cost
            back[i][j] = best_action
        # end for
    # end for

    # Reconstruct the optimal alignment.
    alignment: List[Tuple[int, int]] = []

    i, j = n, m

    while i > 0 or j > 0:
        action = back[i][j]

        if action == "match":
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == "skip_a":
            i -= 1
        elif action == "skip_b":
            j -= 1
        else:
            raise RuntimeError(
                f"Cannot reconstruct alignment at state ({i}, {j})."
            )
        # end if
    # end while

    alignment.reverse()

    return alignment


def align_tokens(g_sent: List[List[str]], t_sent: List[List[str]], t_name: str) -> Dict[int, int]:
    g_tokens: List[str] = []

    for i in range(len(g_sent)):
        word = g_sent[i][1]
        g_tokens.append(word)
    # end for

    t_tokens: List[str] = []

    for i in range(len(t_sent)):
        word = t_sent[i][1]
        t_tokens.append(word)
    # end for

    gt_alignments = _align_sequences_dynamically(list_a=g_tokens, list_b=t_tokens)
    gt_align: Dict[int, int] = {}

    for i, j in gt_alignments:
        gt_align[i] = j
    # end for

    return gt_align


def align_sentences(gs_conllu: List[Tuple[List[str], List[List[str]], List[int]]],
                    tl_conllu: List[Tuple[List[str], List[List[str]], List[int]]],
                    tl_name: str) -> Dict[int, int]:

    def get_text_from_comments(comments: List[str]) -> str:
        for cmnt in comments:
            if cmnt.startswith('# text = '):
                cmnt = cmnt.replace('# text = ', '')
                return cmnt
            # end if
        # end for

        return ''
    # end def

    gs_sentences: List[str] = []

    for i in range(len(gs_conllu)):
        sent_text = get_text_from_comments(comments=gs_conllu[i][0])
        gs_sentences.append(sent_text)
    # end for

    tl_sentences: List[str] = []

    for i in range(len(tl_conllu)):
        sent_text = get_text_from_comments(comments=tl_conllu[i][0])
        tl_sentences.append(sent_text)
    # end for

    gs_tl_alignments = _align_sequences_dynamically(list_a=gs_sentences, list_b=tl_sentences)
    gt_align: Dict[int, int] = {}

    for i, j in gs_tl_alignments:
        gt_align[i] = j
    # end for

    print(f'GS has [{len(gs_conllu)}] sentences; TOOL [{tl_name}] has [{len(tl_conllu)}] sentences', file=sys.stderr)
    print(
        f'GS and TOOL [{tl_name}] have [{len(gt_align)}] sentences in common', file=sys.stderr)

    return gt_align


def mc_nemar_comparison(gs_conllu: List[Tuple[List[str], List[List[str]], List[int]]],
                        t1_conllu: List[Tuple[List[str], List[List[str]], List[int]]],
                        t2_conllu: List[Tuple[List[str], List[List[str]], List[int]]],
                        gs_t1_sent_align: Dict[int, int], gs_t2_sent_align: Dict[int, int],
                        t1_name: str, t2_name: str) -> float:
    # - n11: both correct
    # - n00: both wrong
    # - n10: T1 correct, T2 wrong
    # - n01: T1 wrong, T2 correct
    measures = {
        'lemma': {'n11': 0, 'n00': 0, 'n10': 0, 'n01': 0},
        'upos': {'n11': 0, 'n00': 0, 'n10': 0, 'n01': 0},
        'xpos': {'n11': 0, 'n00': 0, 'n10': 0, 'n01': 0},
        'uas': {'n11': 0, 'n00': 0, 'n10': 0, 'n01': 0},
        'las': {'n11': 0, 'n00': 0, 'n10': 0, 'n01': 0}
    }

    def update_ns(n_dict: Dict[str, int], gsv: str, t1v: str, t2v: str):
        if t1v == gsv and t2v == gsv:
            n_dict['n11'] += 1
        elif t1v != gsv and t2v != gsv:
            n_dict['n00'] += 1
        elif t1v == gsv and t2v != gsv:
            n_dict['n10'] += 1
        elif t1v != gsv and t2v == gsv:
            n_dict['n01'] += 1
        # end if
    # end def

    for i in range(len(gs_conllu)):
        gs_sent = gs_conllu[i][1]

        if i in gs_t1_sent_align and i in gs_t2_sent_align:
            i1 = gs_t1_sent_align[i]
            i2 = gs_t2_sent_align[i]
            t1_sent = t1_conllu[i1][1]
            t2_sent = t2_conllu[i2][1]
            gs_t1_tok_align = align_tokens(g_sent=gs_sent, t_sent=t1_sent, t_name=t1_name)
            gs_t2_tok_align = align_tokens(g_sent=gs_sent, t_sent=t2_sent, t_name=t2_name)

            for j in range(len(gs_sent)):
                if j in gs_t1_tok_align and j in gs_t2_tok_align:
                    j1 = gs_t1_tok_align[j]
                    j2 = gs_t2_tok_align[j]

                    gsv = gs_sent[j][2]
                    t1v = t1_sent[j1][2]
                    t2v = t2_sent[j2][2]
                    update_ns(measures['lemma'], gsv, t1v, t2v)

                    gsv = gs_sent[j][3]
                    t1v = t1_sent[j1][3]
                    t2v = t2_sent[j2][3]
                    update_ns(measures['upos'], gsv, t1v, t2v)

                    gsv = gs_sent[j][4]
                    t1v = t1_sent[j1][4]
                    t2v = t2_sent[j2][4]
                    update_ns(measures['xpos'], gsv, t1v, t2v)

                    gsv = gs_sent[j][6]
                    t1v = t1_sent[j1][6]
                    t2v = t2_sent[j2][6]
                    update_ns(measures['uas'], gsv, t1v, t2v)

                    gsv = gs_sent[j][6] + '/' + gs_sent[j][7]
                    t1v = t1_sent[j1][6] + '/' + t1_sent[j1][7]
                    t2v = t2_sent[j2][6] + '/' + t2_sent[j2][7]
                    update_ns(measures['las'], gsv, t1v, t2v)
                # end if j in both alignments
            # end for all tokens j in GS
        # end if i in both sentence alignments
    # end for

    for meas in ['upos', 'xpos', 'lemma', 'uas', 'las']:
        n00 = measures[meas]['n00']
        n11 = measures[meas]['n11']
        n10 = measures[meas]['n10']
        n01 = measures[meas]['n01']
        table = np.array([[n00, n01],
                          [n10, n11]])
        bundle = mcnemar(table, exact=False, correction=True)
        t1_acc = (n11 + n10) / (n11 + n00 + n10 + n01)
        t2_acc = (n11 + n01) / (n11 + n00 + n10 + n01)
        print(
            f'TOOL [{t1_name}] accuracy is [{t1_acc:.5f}] on measure [{meas}]')
        print(
            f'TOOL [{t2_name}] accuracy is [{t2_acc:.5f}] on measure [{meas}]')
        print(
            f'McNemar\'s statistic is [{bundle.statistic:.5f}], p-value [{bundle.pvalue:.5f}] on measure [{meas}]')
        print(
            f'McNemar\'s counts: [n11 = {n11}], [n00 = {n00}], [n10 = {n10}], [n01 = {n01}] on measure [{meas}]')
        print()
    # end for


def get_tool_name(file_name: str) -> str:
    file_path = Path(file_name)
    tool_name = file_path.name.replace('.conllu', '')
    lpi = tool_name.rfind('.')
    
    return tool_name[lpi + 1:]


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python compare_conllu.py <file.gold.conllu> <file.tool1.conllu> <file.tool2.conllu>")
        sys.exit(1)
    # end if

    gold = read_conllu_file(input_file=sys.argv[1])
    tool1 = read_conllu_file(input_file=sys.argv[2])
    tool1_name = get_tool_name(file_name=sys.argv[2])
    tool2 = read_conllu_file(input_file=sys.argv[3])
    tool2_name = get_tool_name(file_name=sys.argv[3])

    gold_tool1_align = align_sentences(gs_conllu=gold, tl_conllu=tool1, tl_name=tool1_name)
    gold_tool2_align = align_sentences(gs_conllu=gold, tl_conllu=tool2, tl_name=tool2_name)
    print()
    mc_nemar_comparison(gs_conllu=gold, t1_conllu=tool1, t2_conllu=tool2,
                        gs_t1_sent_align=gold_tool1_align, gs_t2_sent_align=gold_tool2_align,
                        t1_name=tool1_name, t2_name=tool2_name)
