import numpy as np


def voice_leading_efficiency(sequence):
    ps = sequence.pitches_as_array()
    return np.absolute(ps[1:] - ps[:-1]).sum(axis=1).mean()


def no_movement(sequence):
    ps = sequence.pitches_as_array()
    diffs = ps[1:] - ps[:-1]
    return (diffs.sum(axis=1) == 0).sum()


def max_center_from_start(sequence):
    ps = sequence.pitches_as_array()
    diffs = ps - ps[0]
    return np.median(diffs, axis=1).max()
