import mido
import numpy as np
from time import sleep
from scout.voicer import Voicer
import itertools
from collections import defaultdict


def is_ordered(seq):
    last = seq[0]
    for val in seq[1:]:
        if val < last:
            return False
        last = val
    return True


def is_lte_octave_span(seq):
    if seq[-1] - seq[0] <= 12:
        return True
    else:
        return False


def sum_in_range(seq):
    seq_sum = sum(seq)
    if 0 <= seq_sum < 12:
        return True
    else:
        return False


def generate_pitch_class(n=2):
    pcs = set()
    ranges = [range(-12, 12) for x in range(n)]
    potential_pcs = itertools.product(*ranges)
    for item in potential_pcs:
        pc = tuple(item)
        if is_ordered(pc) and is_lte_octave_span(pc) and sum_in_range(pc):
            pcs.add(pc)
    return pcs


def filter_pitch_class(pcs):
    new_pcs = set()
    key_1ov = np.array([0, 2, 4, 5, 7, 9, 11])
    key_2ov = np.concatenate([key_1ov, key_1ov - 12])
    key = set(key_2ov.tolist())
    for pc in pcs:
        if pc_in_key(pc,
                     key) and has_no_duplicates(pc) and max_interval_lte(pc):
            new_pcs.add(pc)
    return new_pcs


def normalize_class_nums(pc):
    return tuple([x + 12 if x < 0 else x for x in pc])


def pc_in_key(pc, key):
    return set(pc) < key


def has_no_duplicates(pc):
    return len(set(normalize_class_nums(pc))) == len(pc)


def max_interval_lte(pc, interval=6):
    normalized = normalize_class_nums(pc)
    pitches = np.concatenate([np.array(normalized), [normalized[0]]])
    pitches[-1] = pitches[-1] + 12
    intervals = pitches[1:] - pitches[-1:]
    return intervals.max() < interval


def pitches_to_pitch_class(pitches):
    pc = [p % 12 for p in pitches]
    pc.sort()
    while sum(pc) >= 12:
        pc = [pc[-1] - 12] + pc[:-1]
    assert is_ordered(pc) and is_lte_octave_span(pc) and sum_in_range(pc)
    return tuple(pc)


def generate_voicings(n, pitches, pitch_classes):
    voicing_by_pitch_class = defaultdict(set)
    ranges = [pitches for x in range(n)]
    potential_voicing = itertools.product(*ranges)
    for item in potential_voicing:
        v = tuple(item)
        if is_ordered(v):
            pc = pitches_to_pitch_class(v)
            if pc in pitch_classes:
                voicing_by_pitch_class[pc].add(v)
    output = dict()
    for pc, pitch_set in voicing_by_pitch_class.items():
        output[pc] = np.stack([np.array(p) for p in pitch_set])
    return output


def main():
    outport = mido.open_output('IAC Driver Bus 1')
    pitch_classes = np.array([[0, 4, 7], [-3, 2, 5], [-1, 2, 7]])
    v = Voicer(root=60, start=[0, 4, 7])
    chords = v.from_pitch_class(pitch_classes)

    for chord in chords:
        for note in chord:
            msg = mido.Message('note_on', note=int(note), velocity=100)
            outport.send(msg)
        sleep(1)
        for note in chord:
            outport.send(mido.Message('note_off', note=int(note)))
    outport.close()


def play_chords(chords):
    outport = mido.open_output('IAC Driver Bus 1')
    for chord in chords:
        for note in chord:
            msg = mido.Message('note_on', note=int(note), velocity=100)
            outport.send(msg)
        sleep(1)
        for note in chord:
            outport.send(mido.Message('note_off', note=int(note)))
    outport.close()


if __name__ == '__main__':
    pcs = generate_pitch_class(n=3)
    pcs = filter_pitch_class(pcs)
    voicings = generate_voicings(3, range(60, 96), pcs)
    v = Voicer([60, 64, 67], voicings)
    classes = [[0, 4, 7], [-3, 2, 5], [-1, 2, 7], [0, 4, 7]]
    chords = [v.from_pitch_class(p) for p in classes]
    play_chords(chords)
