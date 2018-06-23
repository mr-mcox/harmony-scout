from old_scout import ScaleSpace, PitchSpace
import numpy as np
from numpy.testing import assert_equal

def test_pitch_values():
    intervals = np.array([0, 2, 4, 5, 7, 9, 11])
    ss = ScaleSpace(root=2, intervals=intervals)
    scale_values = np.array([0, 7, 8])
    expected = np.array([2,14, 16])
    assert_equal(ss.pitch_values(PitchSpace(), scale_values), expected)