from scout.population import Creature, Population
from scout.modules import Judge
from scout.sequencer import Sequencer
import pytest
from random import random
import numpy as np


@pytest.fixture
def simple_pop():
    evolve_params = {"fill": {"target_n": 10}, "cull": {"target_n": 5}}
    p = Population(evolve_params=evolve_params, creature_class=Creature)
    return p


def test_fill_population(simple_pop):
    p = simple_pop
    p.fill()
    assert len(p.creatures) == p.evolve_params["fill"]["target_n"]


def test_cull_population(simple_pop):
    p = simple_pop
    p.fill()
    p.cull()
    assert len(p.creatures) == p.evolve_params["cull"]["target_n"]


class SumJudge(Judge):
    def evaluate(self, phenotype):
        return sum(phenotype)


class RandomCreature(Creature):
    def from_random(self):
        self._phenotype = [random()]


def test_percentile_fitness():
    evolve_params = {"fill": {"target_n": 100}}
    j = SumJudge(sequencer=Sequencer())
    creature_params = {"judges": [j]}
    p = Population(
        creature_class=RandomCreature,
        evolve_params=evolve_params,
        creature_params=creature_params,
    )
    p.fill()
    scores = [c.fitness() for c in p.creatures]
    score_ptile = np.percentile(scores, 90)
    assert score_ptile > 0
    assert p.fitness_ptile(0.9) == score_ptile


class SelfImprovingCreature(Creature):
    def from_random(self, shape):
        self._phenotype = 0

    def from_mutation(self, gene):
        self._phenotype = 1

    def from_crossover(self, genes):
        self._phenotype = 2
