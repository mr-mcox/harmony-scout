from scout.modules import Rhythm, Seq, Consonances
from scout.sequencer import Sequencer
import numpy as np
import pytest


def test_rhythm_out():
    m = Rhythm(sequencer=Sequencer(), params={"durations": [1, 2]})
    outs = list()
    for i in range(6):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [1, 1, 0, 1, 1, 0]


def test_sequencer_out():
    m = Seq(sequencer=Sequencer(), params={"states": [0, 1]})
    inputs = [1, 0, 1]
    outs = list()
    for i in inputs:
        m.input["trigger"] = i
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [0, 0, 1]


@pytest.mark.parametrize(
    "pitches, weight, score", [([0, 7], 1, 1.5), ([-12, 0], 1, 2), ([0, 0], 0, 0)]
)
def test_consonance_score(pitches, weight, score):
    m = Consonances(sequencer=Sequencer(), params={"class_value": [(0, 1), (7, 0.5)]})
    m.input["weight"] = weight
    m.input["trigger"] = 1
    m.resolve_step()
    pitch_class = np.array([pitches])
    assert m.evaluate(pitch_class) == score
