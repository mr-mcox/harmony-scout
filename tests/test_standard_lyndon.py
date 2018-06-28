import numpy as np
from scout.lyndon import standard_lyndon


def test_lyndon_order():
    c_maj = np.array([0, 4, 7])
    c_min = np.array([0, 3, 7])
    d_maj = np.array([-3, 2, 6])
    assert standard_lyndon(c_maj) == standard_lyndon(d_maj)
    assert standard_lyndon(c_maj) != standard_lyndon(c_min)
