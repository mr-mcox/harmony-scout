from old_scout.helpers import is_ordered, spans_lte_octave, sum_in_range
import pytest


@pytest.mark.parametrize("seq, expected", [((0, 1), True), ((1, 0), False)])
def test_is_ordered(seq, expected):
    assert is_ordered(seq) is expected


@pytest.mark.parametrize(
    "seq, expected", [((0, 11), True), ((-11, 0), True), ((0, 12), True), ((0, 13), False)]
)
def test_spans_lte_octave(seq, expected):
    assert spans_lte_octave(seq) is expected


@pytest.mark.parametrize(
    "seq, expected", [((0, 11), True), ((-12, 12), True), ((-12, 0), False)]
)
def test_sum_in_range(seq, expected):
    assert sum_in_range(seq, left=0, right=12) is expected
