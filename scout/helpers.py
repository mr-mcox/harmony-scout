def is_ordered(l):
    """Is the iterable ordered?"""
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1))


def spans_lte_octave(seq):
    """Is the last item in the sequence ≤ an octave span from the first?"""
    return seq[-1] - seq[0] <= 12


def sum_in_range(seq, left=0, right=12):
    return left <= sum(seq) < right
