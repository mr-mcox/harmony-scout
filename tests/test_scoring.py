import numpy as np
import attr
from scout.scoring import voice_leading_efficiency, no_movement


@attr.s
class MockSequence:

    pitch_array = attr.ib(default=np.array([]))

    def pitches_as_array(self):
        return self.pitch_array


def test_voice_leading_efficiency():
    pitches = np.array([[0, 1, 2], [0, 2, 2], [0, 2, 2]])
    seq = MockSequence(pitches)
    assert voice_leading_efficiency(seq) == 0.5

def test_no_movement():
    pitches = np.array([[0, 1, 2], [0, 2, 2], [0, 2, 2]])
    seq = MockSequence(pitches)
    assert no_movement(seq)== 1
