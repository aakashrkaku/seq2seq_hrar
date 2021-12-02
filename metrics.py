import numpy as np

def printChanges(s1, s2, dp, print_true=False):
    i = len(s1)
    j = len(s2)
    deletions = 0
    insertions = 0
    substitutions = 0

    subs_for_margin = 0
    subs_for_middle = 0

    ins_for_margin = 0
    ins_for_middle = 0

    dels_for_margin = 0
    dels_for_middle = 0

    rt_margin = 0
    lt_margin = 0

    # Check till the end
    while (i > 0 and j > 0):
        #         print('i=',i)
        #         print('j=',j)
        #         print('==')

        # If characters are same
        if s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
            if (s2[j - 1] == s2[0]):
                lt_margin = 1
            if (s2[j - 1] == s2[-1]):
                rt_margin = 1



        # Replace
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            if print_true:
                print("change", s1[i - 1],
                      "to", s2[j - 1])
            if (s2[j - 1] == s2[-1]) or (s2[j - 1] == s2[0]):
                subs_for_margin += 1
                if s2[j - 1] == s2[-1]:
                    rt_margin = 1
                else:
                    lt_margin = 1
            else:
                subs_for_middle += 1
            j -= 1
            i -= 1
            substitutions += 1

        # Add
        elif dp[i][j] == dp[i][j - 1] + 1:
            if print_true:
                print("Add", s2[j - 1])
            if (s2[j - 1] == s2[-1]) or (s2[j - 1] == s2[0]):
                ins_for_margin += 1
                if s2[j - 1] == s2[-1]:
                    rt_margin = 1
                else:
                    lt_margin = 1
            else:
                ins_for_middle += 1
            j -= 1
            insertions += 1

        # Delete
        elif dp[i][j] == dp[i - 1][j] + 1:
            if print_true:
                print("Delete", s1[i - 1])
            i -= 1
            deletions += 1
            if (lt_margin == 0) and (rt_margin == 0):
                dels_for_margin += 1
            elif (lt_margin == 1) and (rt_margin == 1):
                dels_for_margin += 1
            else:
                dels_for_middle += 1


        elif i == 0 and j == 0:
            break

    if i == 0:
        for k in range(j):
            if print_true:
                print("Add", s2[k])
            insertions += 1
            if (s2[k] == s2[-1]) or (s2[k] == s2[0]):
                ins_for_margin += 1
                if s2[k] == s2[-1]:
                    rt_margin = 1
                else:
                    lt_margin = 1
            else:
                ins_for_middle += 1
    if j == 0:
        for k in range(i):
            if print_true:
                print("Delete", s1[k])
            deletions += 1
            if (lt_margin == 0) and (rt_margin == 0):
                dels_for_margin += 1
            elif (lt_margin == 1) and (rt_margin == 1):
                dels_for_margin += 1
            else:
                dels_for_middle += 1

    return insertions, deletions, substitutions, ins_for_margin, ins_for_middle, dels_for_margin, dels_for_middle, subs_for_margin, subs_for_middle
# Function to compute the DP matrix
def editDP(s1, s2, print_true=False):
    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for i in range(len2 + 1)]
          for j in range(len1 + 1)]

    # Initilize by the maximum edits possible
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

        # Compute the DP Matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            # If the characters are same
            # no changes required
            if s2[j - 1] == s1[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

                # Minimum of three operations possible
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],
                                   dp[i - 1][j - 1],
                                   dp[i - 1][j])

                # Print the steps
    #     print(dp[len1][len2])
    #     print(np.array(dp))
    i, d, s, i_mr, i_mi, d_mr, d_mi, s_mr, s_mi = printChanges(s1, s2, dp, print_true)
    return i, d, s, i_mr, i_mi, d_mr, d_mi, s_mr, s_mi

def get_fdr_tpr(df):
    ins = []
    dels = []
    subs = []
    pred_len = []
    for i in range(df.shape[0]):
        out = editDP(df['Pred'].iloc[i], df['Gt'].iloc[i])
        ins.append(out[0])
        dels.append(out[1])
        subs.append(out[2])
        pred_len.append(len(df['Pred'].iloc[i]))
    df['ins'] = ins
    df['dels'] = dels
    df['subs'] = subs
    df['len_pred'] = pred_len
    out = df
    out['FDR'] = (out['dels'] + out['subs']) / out['len_pred']
    out['TPR'] = 1 - (out['ins'] + out['subs']) / out['len_gt']
    mean_fdr = np.mean(out['FDR'])
    mean_tpr = np.mean(out['TPR'])

    return mean_fdr, mean_tpr, df


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    #     print(P)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    #     print(Y)
    return levenstein(P, Y, norm)



def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                #                 s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                #                 distance[cur_row_idx][j] = min(s_num, i_num, d_num)
                distance[cur_row_idx][j] = min(i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence. already list
    :type reference: list
    :param hypothesis: The hypothesis sentence. already list
    :type hypothesis: list
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    # if ignore_case == True:
    #     reference = reference.lower()
    #     hypothesis = hypothesis.lower()

    # ref_words = reference.split(delimiter)
    # hyp_words = hypothesis.split(delimiter)
    ref_words = reference
    hyp_words = hypothesis

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

