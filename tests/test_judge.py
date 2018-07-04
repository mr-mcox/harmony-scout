from scout.modules import Judge
from scout.sequencer import Sequencer
import numpy as np
from numpy.testing import assert_array_equal


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
