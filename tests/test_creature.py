from scout.population import (
    PitchClassCreature,
    pitch_classes_with_pitch,
    Creature,
    CreatureFactory,
)
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.random import RandomState
import attr


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
    gene = np.array([before])
    c = PitchClassCreature(gene=gene)
    res = c.conform_genotype(gene)
    exp = np.array([after])
    assert_almost_equal(res, exp)


def test_conform_phenotype():
    valid_pheno = np.array([[0, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 3]])
    gene = np.array([[0.9, 1.1, 0.9], [0.1, 0.9, 0.8]]) / 12
    c = PitchClassCreature(valid_phenotypes=valid_pheno, gene=gene)
    res = c.conform_phenotype(gene)
    exp = np.array([[1, 1, 1], [0, 1, 1]])
    assert_almost_equal(res, exp)


def test_chords_with_pitches():
    pitches = [0, 2, 5, 7]
    res = pitch_classes_with_pitch(pitches, n=2)
    assert res.shape[1] == 2
    assert len(np.unique(res, axis=0)) == res.shape[0]
    chords = set()
    for i in range(res.shape[0]):
        chords.add(tuple(res[i].tolist()))
    assert (0, 5) in chords


def test_pitch_class_from_random():
    rand = RandomState(42)
    cf = CreatureFactory(
        creature_class=PitchClassCreature, random_state=rand, gene_shape=(10, 3)
    )
    c = cf.from_random()
    assert c.genotype.shape == (10, 3)


def test_pitch_class_from_mutation():
    rand = RandomState(43)
    cf = CreatureFactory(
        creature_class=PitchClassCreature, random_state=rand, gene_shape=(20, 3)
    )
    c1 = cf.from_random()
    c2 = cf.from_mutation(c1)
    diff = np.abs(c1.genotype - c2.genotype).sum()
    assert 0 < diff < 1


def test_pitch_class_from_crossover():
    rand = RandomState(43)
    cf = CreatureFactory(
        creature_class=PitchClassCreature, random_state=rand, gene_shape=(20, 3)
    )
    c1 = cf.from_random()
    c2 = cf.from_random()
    c3 = cf.from_crossover([c1, c2])
    diff = np.stack(
        [
            np.abs(c3.genotype - c1.genotype).sum(axis=1),
            np.abs(c3.genotype - c2.genotype).sum(axis=1),
        ],
        axis=1,
    )
    assert (diff.max(axis=0) > 0).all()  # Don't just match a single one
    assert (diff.min(axis=1) == 0).all()


@attr.s
class PartialJudge:
    score = attr.ib(default=1)

    def evaluate(self, phenotype):
        return self.score


def test_creature_fitness():
    j1 = PartialJudge(score=1)
    j2 = PartialJudge(score=0.5)
    c = Creature(judges=[j1, j2], gene=None)
    c._proto_gene = 0  # To not trip the error
    assert c.fitness() == 1.5
