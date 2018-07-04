from scout.sequencer import Sequencer
from scout.modules import Judge


class MyJudge(Judge):
    pass


def test_sequencer_registers_judges():
    s = Sequencer()
    MyJudge(sequencer=s, level="pitch_class")
    assert len(s.for_level["pitch_class"]) == 1
