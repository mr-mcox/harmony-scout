from scout.voicer import Voicer
import numpy as np
from numpy.testing import assert_array_equal

def test_voicer():
    pitch_class = np.array([[0, 3]])
    v = Voicer(root=60, start=[0, 2])
    assert_array_equal( v.from_pitch_class(pitch_class), np.array([[60, 63]]))