import attr
import numpy as np


@attr.s
class PitchSpace:

    steps_in_octave = attr.ib(default=12)


@attr.s
class ScaleSpace:

    root = attr.ib(default=0)
    intervals = attr.ib(default=np.array([0, 2, 4, 5, 7, 9, 11]))

    @property
    def n_steps(self):
        return len(self.intervals)

    def pitch_values(self, pitch_space, values):
        octave, scale_class = np.divmod(values, self.n_steps)
        pitches = self.root + self.intervals[scale_class] + octave * pitch_space.steps_in_octave
        return pitches
