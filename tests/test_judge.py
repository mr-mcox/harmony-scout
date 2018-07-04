from scout.modules import Judge, AuthenticCadence
from scout.sequencer import Sequencer
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal


class MyJudge(Judge):
    default_params = {"trigger": 1}
    input_parameters = ["trigger"]

    def array_vals(self):
        return {"out": (0, 1)}


def test_judge_has_array_output():
    j = MyJudge(sequencer=Sequencer())
    j.resolve_step()
    exp = np.array([[0, 1]])
    assert_array_equal(j.array_output["out"], exp)

    j.resolve_step()
    exp = np.array([[0, 1], [0, 1]])
    assert_array_equal(j.array_output["out"], exp)


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
    r = AuthenticCadence(scale=[0, 2, 4, 5, 7, 9, 11], n=n, sequencer=Sequencer())
    roles, strengths = r.functional_role(np.array([pitch_classes]))
    assert roles[0] == role
    assert abs(strengths[0] - 1) < 0.001


def test_strength_lt_1():
    pitch_classes = np.array([0, 5, 7])
    r = AuthenticCadence(scale=[0, 2, 4, 5, 7, 9, 11], n=3, sequencer=Sequencer())
    roles, strengths = r.functional_role(np.array([pitch_classes]))
    assert roles[0] == 0
    assert strengths[0] < 1


def test_multiple_functional_role():
    pitch_classes = np.array([[0, 4, 7], [-3, 2, 5]])
    r = AuthenticCadence(scale=[0, 2, 4, 5, 7, 9, 11], n=3, sequencer=Sequencer())
    roles, strengths = r.functional_role(np.array(pitch_classes))
    assert_array_equal(roles, np.array([0, 1]))
    assert_almost_equal(strengths, np.array([1, 1]))
