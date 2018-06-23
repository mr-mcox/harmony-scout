from scout.modules import Rhythm, Seq
from scout.sequencer import Sequencer


def test_rhythm_out():
    m = Rhythm(sequencer=Sequencer(), params={"durations": [1, 2]})
    outs = list()
    for i in range(6):
        m.resolve_step()
        outs.append(m.output["out"])
    assert outs == [1, 1, 0, 1, 1, 0]

def test_sequencer_out():
    m = Seq(sequencer=Sequencer(), params={'states': [0, 1]})
    inputs = [1, 0, 1]
    outs = list()
    for i in inputs:
        m.input['trigger'] = i
        m.resolve_step()
        outs.append(m.output['out'])
    assert outs == [0, 0, 1]