import mido
import numpy as np
from scout.spaces import ScaleSpace
from scout.chord import ExactScaleCT, RandomWalkCT, CopyCT
from scout.scoring import voice_leading_efficiency, no_movement, max_center_from_start
from scout.sequence import SequenceTemplate
from time import sleep
from operator import itemgetter


def main():
    outport = mido.open_output('IAC Driver Bus 1')

    scale = ScaleSpace(root=60)
    c1 = ExactScaleCT(scale_space=scale, scale_values=np.array([0, 2, 4, 8]))
    c2 = RandomWalkCT(scale_space=scale)
    c3 = CopyCT()
    chord_templates = [c1, c2, c2, c2, c2, c3]
    scoring_rules = [(-1, voice_leading_efficiency), (-20, no_movement),
                     (1, max_center_from_start)]
    st = SequenceTemplate(
        chord_templates=chord_templates, scoring_rules=scoring_rules)
    st.add_link(0, 1)
    st.add_link(1, 2)
    st.add_link(2, 3)
    st.add_link(3, 4)
    st.add_link(0, 5)
    sequences = [st.generate() for i in range(1000)]
    seq_with_score = [(s, s.score()) for s in sequences]
    seq_sorted = [s[0] for s in sorted(seq_with_score, key=itemgetter(1))]
    for si in seq_sorted[-5:]:
        print(si.score())

        chords = si.chords
        for chord in chords:
            for note in np.nditer(chord.pitches):
                msg = mido.Message('note_on', note=int(note), velocity=100)
                outport.send(msg)
            sleep(1)
            for note in np.nditer(chord.pitches):
                outport.send(mido.Message('note_off', note=int(note)))
        sleep(1)
    outport.close()


if __name__ == '__main__':
    main()
