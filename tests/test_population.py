from scout.population import Creature, Population, CreatureFactory, population_factory
from scout.modules import Judge
from scout.sequencer import Sequencer
import pytest
import numpy as np
from numpy.random import RandomState


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
    def from_random(self):
        return self.creature_class([0], judges=self.judges)

    def from_mutation(self, gene):
        return self.creature_class([1], judges=self.judges)

    def from_crossover(self, genes):
        return self.creature_class([2], judges=self.judges)


def test_evolve_improves():
    random_state = RandomState(3842)
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
