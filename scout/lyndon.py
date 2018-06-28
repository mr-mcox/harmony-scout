import numpy as np


def lyndon_words(s, n):
    # From https://gist.github.com/dvberkel/1950267
    w = [-1]  # set up for first increment
    while w:
        w[-1] += 1  # increment the last non-z symbol
        yield w
        m = len(w)
        while len(w) < n:  # repeat word to fill exactly n syms
            w.append(w[-m])
        while w and w[-1] == s - 1:  # delete trailing z's
            w.pop()


def lyndon_by_order(s, n):
    return {tuple(w): i for i, w in enumerate(lyndon_words(s, n))}


def cfl_breakpoints(s):
    # From https://gist.github.com/dvberkel/1950267
    """Find starting positions of Chen-Fox-Lyndon decomposition of s.
    The decomposition is a set of Lyndon words that start at 0 and
    continue until the next position. 0 itself is not output, but
    the final breakpoint at the end of s is. The argument s must be
    of a type that can be indexed (e.g. a list, tuple, or string).
    The algorithm follows Duval, J. Algorithms 1983, but uses 0-based
    indexing rather than Duval's choice of 1-based indexing."""
    k = 0
    while k < len(s):
        i, j = k, k + 1
        while j < len(s) and s[i] <= s[j]:
            i = (s[i] == s[j]) and i + 1 or k  # Python cond?yes:no syntax
            j += 1
        while k < i + 1:
            k += j - i
            yield k


def cfl(s):
    # From https://gist.github.com/dvberkel/1950267
    """Decompose s into Lyndon words according to the Chen-Fox-Lyndon theorem.
    The arguments are the same as for cfl_breakpoints but the
    return values are subsequences of s rather than indices of breakpoints."""
    old = 0
    for k in cfl_breakpoints(s):
        yield s[old:k]
        old = k


def smallest_rotation(s):
    # From https://gist.github.com/dvberkel/1950267
    """Find the rotation of s that is smallest in lexicographic order.
    Duval 1983 describes how to modify his algorithm to do so but I think
    it's cleaner and more general to work from the ChenFoxLyndon output."""
    prev, rep = None, 0
    for w in cfl(s + s):
        if w == prev:
            rep += 1
        else:
            prev, rep = w, 1
        if len(w) * rep == len(s):
            return w * rep
    raise Exception("Reached end of factorization with no shortest rotation")


def standard_lyndon(pitch_classes, octave_size=12):
    extended = np.concatenate([pitch_classes, [pitch_classes[0] + octave_size]])
    intervals = extended[1:] - extended[:-1]
    onsets = list()
    for i in intervals.tolist():
        if i > 0:
            onsets.append(1)
            onsets.extend([0] * (i - 1))
    return smallest_rotation(onsets)


if __name__ == "__main__":
    print(smallest_rotation([1, 0, 1, 1, 0]))
