from scout.population import PitchClassCreature
import pytest
import numpy as np
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize(
    "before,after",
    [
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
        ([0.2, 0.1, 0.3], [0.1, 0.2, 0.3]),
        ([0.1, 0.2, -0.7], [0.1, 0.2, 0.3]),
        ([0.1, 0.2, 0.8], [-0.2, 0.1, 0.2]),
        ([0.4, 0.8, 0.8], [-0.2, -0.2, 0.4]),
    ],
)
def test_conform_genotype(before, after):
    c = PitchClassCreature()
    gene = np.array([before])
    res = c.conform_genotype(gene)
    exp = np.array([after])
    assert_almost_equal(res, exp)


def test_conform_phenotype():
    valid_pheno = np.array([[0, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 3]])
    gene = np.array([[0.9, 1.1, 0.9], [0.1, 0.9, 0.8]]) / 12
    c = PitchClassCreature()
    res = c.conform_phenotype(gene, valid_pheno)
    exp = np.array([[1, 1, 1], [0, 1, 1]])
    assert_almost_equal(res, exp)


def test_chords_with_pitches():
    pitches = [0, 5, 7]
    c = PitchClassCreature()
    res = c.pitch_classes_with_pitch(pitches, n=2)
    assert res.shape[1] == 2
    assert len(np.unique(res, axis=0)) == len(res)
