from scout.population import (
    Creature,
    Population,
    CreatureFactory,
    population_factory,
    VoicingCreature,
)
from scout.modules import Judge
from scout.sequencer import Sequencer
import pytest
import numpy as np
from numpy.random import RandomState
from numpy.testing import assert_array_equal


@pytest.fixture
def simple_pop():
    evolve_params = {"fill": {"target_n": 10}, "cull": {"target_n": 5}}
    cf = CreatureFactory(creature_class=Creature)
    p = Population(evolve_params=evolve_params, creature_factory=cf)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._genotype = np.random.random(size=1)


def test_percentile_fitness():
    evolve_params = {"fill": {"target_n": 100}}
    j = SumJudge(sequencer=Sequencer())
    cf = CreatureFactory(creature_class=RandomCreature, judges=[j])
    p = Population(evolve_params=evolve_params, creature_factory=cf)
    p.fill()
    scores = [c.fitness() for c in p.creatures]
    score_ptile = np.percentile(scores, 90)
    assert score_ptile > 0
    assert p.fitness_ptile(0.9) == score_ptile


class HackedEvolutionCreatureFactory(CreatureFactory):
    def from_random(self, parent=None):
        return self.creature_class([0], judges=self.judges)

    def from_mutation(self, gene, parent=None):
        return self.creature_class([1], judges=self.judges)

    def from_crossover(self, genes, parent=None):
        return self.creature_class([2], judges=self.judges)


def test_evolve_improves():
    random_state = RandomState(842)
    evolve_params = {
        "fill": {"target_n": 20},
        "cull": {"target_n": 10},
        "evolve": {"target_n": 20, "origin_probs": [0.5, 0.3, 0.2]},
    }
    j = SumJudge(sequencer=Sequencer())
    cf = HackedEvolutionCreatureFactory(creature_class=Creature, judges=[j])
    p = Population(
        creature_factory=cf, evolve_params=evolve_params, random_state=random_state
    )
    p.evolve(to_generation=1)
    assert p.generations == 1
    prev_fitness = p.fitness_ptile(0.5)
    p.evolve(to_generation=2)
    assert p.fitness_ptile(0.5) > prev_fitness


def test_evolve_sub_pop():
    cf = CreatureFactory(creature_class=Creature)
    pf = population_factory(Population, creature_factory=cf)
    c = Creature(gene=[0], population_factory=pf)
    c.evolve_sub_population()
    assert c.sub_population.generations == 0


def test_sub_pop_on_evolve():
    sub_creature_factory = CreatureFactory(creature_class=Creature)
    pf = population_factory(Population, creature_factory=sub_creature_factory)
    cf = CreatureFactory(creature_class=Creature, sub_population_factory=pf)
    p = Population(creature_factory=cf)
    p.evolve(to_generation=1)
    assert p.creatures[0].sub_population.generations == 1


def test_sub_population_parent():
    sub_creature_factory = CreatureFactory(creature_class=Creature)
    pf = population_factory(Population, creature_factory=sub_creature_factory)
    cf = CreatureFactory(creature_class=Creature, sub_population_factory=pf)
    p = Population(creature_factory=cf)
    p.evolve(to_generation=1)
    for creature in p.creatures:
        assert creature.sub_population.creatures[0].parent == creature


@pytest.mark.parametrize(
    "pitches,pitch_class", [([0, 1, 2], [0, 1, 2]), ([60, 61, 62], [0, 1, 2])]
)
def test_voicing_creature_voices_for(pitches, pitch_class):
    valid = np.array([pitches])
    vc = VoicingCreature(gene=np.array([0]), valid_phenotypes=valid)
    assert_array_equal(vc.voices_for[tuple(pitch_class)], np.array([pitches]))


def test_multiple_voices_for():
    valid = np.array([[0, 1, 2], [0, 1, 3]])
    vc = VoicingCreature(gene=np.array([0]), valid_phenotypes=valid)
    assert_array_equal(vc.voices_for[(0, 1, 2)], np.array([[0, 1, 2]]))
    assert_array_equal(vc.voices_for[(0, 1, 3)], np.array([[0, 1, 3]]))


def test_voicing_creature_phenotype():
    pc = Creature(np.array([[0, 1, 2], [0, 1, 2]]))
    valid = np.array([[72, 73, 74], [60, 61, 62], [60, 61, 63], [60, 61, 50]])
    start_voice = np.array([60, 63, 65])
    vc = VoicingCreature(
        gene=np.array([0, 0.5]),
        valid_phenotypes=valid,
        parent=pc,
        start_voice=start_voice,
    )
    assert_array_equal(vc.phenotype, np.array([[60, 61, 62], [60, 61, 50]]))
