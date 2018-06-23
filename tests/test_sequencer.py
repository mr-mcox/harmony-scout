from scout.sequencer import Sequencer
from scout.modules import Judge


class MyJudge(Judge):
    level = "pitch_class"


def test_sequencer_registers_judges():
    s = Sequencer()
    MyJudge(sequencer=s)
    assert len(s.for_level["pitch_class"]) == 1
