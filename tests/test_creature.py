from scout.population import PitchClassCreature
import pytest
import numpy as np
from numpy.testing import assert_almost_equal

@pytest.mark.parametrize('before,after',
                         [
                             ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
                             ([0.2, 0.1, 0.3], [0.1, 0.2, 0.3]),
                             ([0.1, 0.2, -0.7], [0.1, 0.2, 0.3]),
                             ([0.1, 0.2, 0.8], [-0.2, 0.1, 0.2]),
                             ([0.4, 0.8, 0.8], [-0.2, -0.2, 0.4]),
                         ])
def test_conform_genotype(before, after):
    c = PitchClassCreature()
    gene = np.array([before])
    res = c.conform_genotype(gene)
    exp = np.array([after])
    assert_almost_equal(res, exp)
