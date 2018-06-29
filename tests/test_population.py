from scout.population import Creature, Population
import pytest


@pytest.fixture
def simple_pop():
    c_params = {"random": {"shape": (1, 1)}}
    evolve_params = {"fill": {"target_n": 10}, "cull": {"target_n": 5}}
    p = Population(
        evolve_params=evolve_params, creature_class=Creature, creature_params=c_params
    )
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
