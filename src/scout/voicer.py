import numpy as np


class Voicer:
    def __init__(self, start, voicings_by_pc):
        self.last = np.array(start)
        self.voicings = voicings_by_pc

    def from_pitch_class(self, pc):
        pc = tuple(pc)
        voicings = self.voicings[pc]
        dist = np.sum(np.square(self.last - voicings), axis=1)
        smallest = np.argmin(dist)
        return voicings[smallest]

