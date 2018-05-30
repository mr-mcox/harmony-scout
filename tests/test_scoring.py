import numpy as np
import attr
from scout.scoring import voice_leading_efficiency, no_movement, max_center_from_start


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
    assert no_movement(seq) == 1


def test_max_distance_from_start():
    pitches = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 2]])
    seq = MockSequence(pitches)
    assert max_center_from_start(seq) == 1
