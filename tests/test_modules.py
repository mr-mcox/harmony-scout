from scout.modules import Rhythm, Seq, Consonances, CadenceDetector, Sawtooth
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal


def test_rhythm_out():
    m = Rhythm(durations=[1, 2, 1])
    outs = list()
    for i in range(4):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [x / 4 for x in [0, 1, 3, 0]]


def test_sequencer_out():
    m = Seq(states=[0, 1])
    outs = list()
    for i in range(3):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [0, 1, 0]


def test_sequencer_from_clock():
    m = Seq(states=[0, 1])
    outs = list()
    clocks = [0, 0.25, 0.5]
    for c in clocks:
        m.input["clock"] = c
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [0, 0, 1]


@pytest.mark.parametrize(
    "pitches, weight, score", [([0, 7], 1, 1.5), ([-12, 0], 1, 2), ([0, 0], 0, 0)]
)
def test_consonance_score(pitches, weight, score):
    m = Consonances(class_value=[(0, 1), (7, 0.5)])
    m.input["weight"] = weight
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
    r = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=n)
    roles, strengths = r.functional_role(np.array([pitch_classes]))
    assert roles[0] == role
    assert abs(strengths[0] - 1) < 0.001


def test_strength_lt_1():
    pitch_classes = np.array([0, 5, 7])
    r = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3)
    roles, strengths = r.functional_role(np.array([pitch_classes]))
    assert roles[0] == 0
    assert strengths[0] < 1


def test_multiple_functional_role():
    pitch_classes = np.array([[0, 4, 7], [-3, 2, 5]])
    r = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3)
    roles, strengths = r.functional_role(np.array(pitch_classes))
    assert_array_equal(roles, np.array([0, 1]))
    assert_almost_equal(strengths, np.array([1, 1]))


@pytest.mark.parametrize(
    "cadence,pitches,expected",
    [
        ("authentic", [[-1, 2, 7], [0, 4, 7]], 1),
        ("authentic", [[-1, 2, 5], [0, 4, 7]], 1),
        ("plagel", [[-3, 2, 5], [0, 4, 7]], 1),
        ("deceptive", [[-1, 2, 7], [-3, 0, 5]], 1),
        ("non_authentic", [[-3, 2, 5], [0, 4, 7]], 1),
        ("non_authentic", [[-1, 2, 7], [-3, 0, 5]], 1),
        ("ascending", [[0, 4, 7], [-1, 2, 7]], 1),
    ],
)
def test_evaluate_authentic_cadence(cadence, pitches, expected):
    m = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3)
    weight_type = f"weight_{cadence}"
    m.input[weight_type] = 1
    m.resolve_step()
    m.input[weight_type] = 0
    m.resolve_step()
    pitches = np.array(pitches)
    assert m.evaluate(pitches) == expected


def test_evaluate_cadence_weights():
    m = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3)
    m.input["weight_authentic"] = 0.5
    m.resolve_step()
    m.input["weight_authentic"] = 0
    m.resolve_step()
    pitches = np.array([[-1, 2, 7], [0, 4, 7]])
    assert m.evaluate(pitches) == 0.5


def test_evaluate_cadence_loop():
    m = CadenceDetector(scale=[0, 2, 4, 5, 7, 9, 11], n=3)
    m.input["weight_authentic"] = 0
    m.resolve_step()
    m.input["weight_authentic"] = 1
    m.resolve_step()
    pitches = np.array([[0, 4, 7], [-1, 2, 7]])
    assert m.evaluate(pitches) == 1


def test_sawtooth_default():
    m = Sawtooth(period=4)
    outs = list()
    for i in range(5):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [-1, -0.5, 0, 0.5, -1]


def test_sawtooth_clock():
    m = Sawtooth(period=1)
    clocks = [0, 0.25, 0.5, 0.75, 0]
    outs = list()
    for c in clocks:
        m.input["clock"] = c
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [-1, -0.5, 0, 0.5, -1]


def test_sawtooth_phase():
    m = Sawtooth(period=4, phase=2)
    outs = list()
    for i in range(5):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [0, 0.5, -1, -0.5, 0]


def test_sawtooth_dc():
    m = Sawtooth(period=4, is_ac=False)
    outs = list()
    for i in range(5):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [0, 0.25, 0.5, 0.75, 0]
