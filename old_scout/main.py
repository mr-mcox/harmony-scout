import mido
import numpy as np
from time import sleep
from old_scout.voicer import Voicer
from old_scout.helpers import is_ordered, spans_lte_octave, sum_in_range
import itertools


def generate_pitch_class(n=2):
    pcs = set()
    ranges = [range(-12, 12) for x in range(n)]
    potential_pcs = itertools.product(*ranges)
    for item in potential_pcs:
        pc = tuple(item)
        if is_ordered(pc) and spans_lte_octave(pc) and sum_in_range(pc):
            pcs.add(pc)
    return pcs


def normalize_class_nums(pc):
    return tuple([x + 12 if x < 0 else x for x in pc])


def max_interval_lte(pc, interval=6):
    normalized = normalize_class_nums(pc)
    pitches = np.concatenate([np.array(normalized), [normalized[0]]])
    pitches[-1] = pitches[-1] + 12
    intervals = pitches[1:] - pitches[-1:]
    return intervals.max() < interval


def pc_in_key(pc, key):
    return set(pc) < key


def has_no_duplicates(pc):
    return len(set(normalize_class_nums(pc))) == len(pc)


def filter_pitch_class(pcs):
    new_pcs = set()
    key_1ov = np.array([0, 2, 4, 5, 7, 9, 11])
    key_2ov = np.concatenate([key_1ov, key_1ov - 12])
    key = set(key_2ov.tolist())
    for pc in pcs:
        if pc_in_key(pc, key) and has_no_duplicates(pc) and max_interval_lte(pc):
            new_pcs.add(pc)
    return new_pcs


def play_chords(chords):
    outport = mido.open_output("IAC Driver Bus 1")
    for i in range(chords.shape[0]):
        chord = chords[i]
        for note in chord:
            msg = mido.Message("note_on", note=int(note), velocity=100)
            outport.send(msg)
        sleep(1)
        for note in chord:
            outport.send(mido.Message("note_off", note=int(note)))
    outport.close()


def parallel_vecs(pcs):
    pcs = np.array(pcs)
    unit = np.ones(pcs.shape[1])
    norm = unit / np.linalg.norm(unit)
    diffs = pcs[1:] - pcs[:-1]
    d_norm = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)
    print(np.dot(d_norm, norm))


def main():
    pitch_classes = generate_pitch_class(n=3)
    pitch_classes = filter_pitch_class(pitch_classes)
    v = Voicer(n_voices=3, pitch_classes=pitch_classes, pitch_choices=range(60, 84))
    classes = np.array([[0, 4, 7], [-1, 2, 5], [-3, 2, 5], [-1, 2, 7], [0, 4, 7]])
    chords = v.voice_pitch_classes(np.array([60, 65, 67]), classes)
    parallel_vecs(classes)
    play_chords(chords)


if __name__ == "__main__":
    main()
