from .helpers import is_ordered, spans_lte_octave, sum_in_range
import itertools
from collections import defaultdict
import numpy as np


def pitches_to_pitch_class(pitches):
    pc = [p % 12 for p in pitches]
    pc.sort()
    while sum(pc) >= 12:
        pc = [pc[-1] - 12] + pc[:-1]
    assert is_ordered(pc) and spans_lte_octave(pc) and sum_in_range(pc)
    return tuple(pc)


class Voicer:
    """Generate voicing of chord"""

    def __init__(self, pitch_classes, pitch_choices, n_voices=4):
        self.pitch_classes = pitch_classes
        self.pitch_choices = pitch_choices
        self.n_voices = n_voices
        self.voicing_options = self.create_voicing_options()

    def create_voicing_options(self):
        """Generate dictionary of voicings by brute force"""
        voicings = defaultdict(set)
        choice_copies = [self.pitch_choices for x in range(self.n_voices)]
        possibilities = itertools.product(*choice_copies)
        for possibility in possibilities:
            v = tuple(possibility)
            pc = pitches_to_pitch_class(v)
            if is_ordered(v) and pc in self.pitch_classes:
                voicings[pc].add(v)
        options = dict()
        for pc, pitch_set in voicings.items():
            options[pc] = np.stack([np.array(p) for p in pitch_set])
        return options

    def from_pitch_class(self, start, pc):
        """
        Generate from pitch class

        Parameters
        ----------
        pc: ndarray
            Matrix of pitch classes - pitch class on axis 1

        Returns
        -------
        ndarray
            Pitches of minimal voicing

        """
        pc = tuple(pc)
        voicings = self.voicings[pc]
        dist = np.sum(np.square(self.last - voicings), axis=1)
        smallest = np.argmin(dist)
        return voicings[smallest]

    def smallest_leading(self, prior, pitch_class):
        """
        Find the smallest leading

        Parameters
        ----------
        prior: ndarray of int (dim=1)
            The previous chord to lead from
        pitch_class: iter of int
            The pitch class of the next chord

        Returns
        -------
        ndarray

        """
        pc = tuple(pitch_class)
        possibles = self.voicing_options[pc]
        distances = np.sum(np.square(prior - possibles), axis=1)
        smallest = np.argmin(distances)
        return possibles[smallest]

    def voice_pitch_classes(self, start, pitch_classes):
        """
        Find the smallest leadings for a set of pitch classes

        Parameters
        ----------
        start: ndarray of int (dim=1)
            The starting chord to lead from
        pitch_classes: ndarray of int (dim=2)
            The pitch classes

        Returns
        -------
        ndarray (dim=2)

        """
        prior = start
        chords = list()
        for i in range(pitch_classes.shape[0]):
            pitch_class = pitch_classes[i]
            chords.append(self.smallest_leading(prior, pitch_class))
        return np.array(chords)
