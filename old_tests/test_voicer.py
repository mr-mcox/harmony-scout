from old_scout.voicer import Voicer
import numpy as np


def pitches_to_pitch_class(pitches):
    pc = [p % 12 for p in pitches]
    pc.sort()
    while sum(pc) >= 12:
        pc = [pc[-1] - 12] + pc[:-1]
    return tuple(pc)


def test_pitch_map():
    pitch_classes = {(0, 7)}
    v = Voicer(n_voices=2, pitch_classes=pitch_classes, pitch_choices=range(60, 72))
    for option in v.voicing_options[(0, 7)]:
        assert len(option) == 2
        assert pitches_to_pitch_class(option) == (0, 7)


def test_smallest_voicing():
    pitch_class = (0, 3, 5)
    pitch_classes = {pitch_class}
    v = Voicer(n_voices=3, pitch_classes=pitch_classes, pitch_choices=range(60, 84))
    prev = np.array([60, 65, 70])
    res = v.smallest_leading(prev, pitch_class)
    assert pitches_to_pitch_class(res) == pitch_class
    options = v.voicing_options[pitch_class]
    assert_smallest(res, prev, options)


def assert_smallest(target, prior, options):
    chosen_distance = sum(target - prior)
    other_distance = options - prior
    other_distance_abs = np.abs(other_distance).sum(axis=1)
    assert (other_distance_abs >= chosen_distance).all()


def test_voicings():
    pc_list = [(0, 3, 5), (0, 2, 6), (-1, 3, 4)]
    pitch_classes = set(pc_list)
    v = Voicer(n_voices=3, pitch_classes=pitch_classes, pitch_choices=range(60, 84))
    prev = np.array([60, 65, 70])
    res = v.voice_pitch_classes(start=prev, pitch_classes=np.array(pc_list))
    prior = prev
    for i in range(res.shape[0]):
        target = res[i]
        options = v.voicing_options[pc_list[i]]
        assert_smallest(target, prior, options)
        prior = target
