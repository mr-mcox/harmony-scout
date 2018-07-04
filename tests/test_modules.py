from scout.modules import Rhythm, Seq, Consonances, CadenceDetector
from scout.sequencer import Sequencer
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal


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


@pytest.mark.parametrize(
    "pitch_classes,n,role",
    [
        ([0, 4, 7], 3, 0),
        ([-3, 2, 5], 3, 1),
        ([-5, 0, 4, 7], 4, 0),
        ([-1, 0, 4, 7], 4, 0),
    ],
)
def test_functional_role(pitch_classes, n, role):
    r = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=n, sequencer=Sequencer())
    roles, strengths = r.functional_role(np.array([pitch_classes]))
    assert roles[0] == role
    assert abs(strengths[0] - 1) < 0.001


def test_strength_lt_1():
    pitch_classes = np.array([0, 5, 7])
    r = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3, sequencer=Sequencer())
    roles, strengths = r.functional_role(np.array([pitch_classes]))
    assert roles[0] == 0
    assert strengths[0] < 1


def test_multiple_functional_role():
    pitch_classes = np.array([[0, 4, 7], [-3, 2, 5]])
    r = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3, sequencer=Sequencer())
    roles, strengths = r.functional_role(np.array(pitch_classes))
    assert_array_equal(roles, np.array([0, 1]))
    assert_almost_equal(strengths, np.array([1, 1]))
