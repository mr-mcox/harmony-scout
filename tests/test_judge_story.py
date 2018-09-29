import pytest
from scout.sequencer import build_modules, Sequencer
import numpy as np


@pytest.fixture
def judge_configs():
    defs = [
        {"type": "rhythm", "params": {"durations": [3, 1]}},
        {
            "type": "seq",
            "params": {"states": [0, 1]},
            "patches": [{"source": {"name": "rhythm"}, "dest": "clock"}],
        },
        {
            "type": "consonances",
            "params": {"class_value": [(0, 1), (7, 0.5)]},
            "patches": [
                {"source": {"name": "seq"}, "dest": "weight"},
            ],
        },
    ]
    return defs


def test_story(judge_configs):
    # Given a dictionary of judge definitions
    # And a sequencer
    s = Sequencer(length=4)
    # A judge factory
    build_modules(configs=judge_configs, sequencer=s)
    # When the sequencer is run
    s.sequence()
    # It has a rule that evaluates a phenotype as expected
    good_phenotype = np.array([3, 0, 1, 7])
    bad_phenotype = np.array([0, 1, 7, 3])
    judge = s.for_level["pitch_class"][0]
    assert judge.evaluate(bad_phenotype) == 0
    assert judge.evaluate(good_phenotype) == 1.5
