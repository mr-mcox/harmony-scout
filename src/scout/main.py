import mido
import numpy as np
from scout.spaces import ScaleSpace
from scout.chord import ExactScaleCT, RandomWalkCT, CopyCT
from scout.sequence import SequenceTemplate
from time import sleep


def main():
    outport = mido.open_output('IAC Driver Bus 1')

    scale = ScaleSpace(root=60)
    c1 = ExactScaleCT(scale_space=scale, scale_values=np.array([0, 2, 4, 8]))
    c2 = RandomWalkCT(scale_space=scale)
    c3 = CopyCT()
    chord_templates = [c1, c2, c2, c3]
    st = SequenceTemplate(chord_templates=chord_templates)
    st.add_link(0, 1)
    st.add_link(1, 2)
    st.add_link(0, 3)
    chords = st.generate().chords
    for chord in chords:
        for note in np.nditer(chord.pitches):
            msg = mido.Message('note_on', note=int(note), velocity=100)
            outport.send(msg)
        sleep(1)
        for note in np.nditer(chord.pitches):
            outport.send(mido.Message('note_off', note=int(note)))
    outport.close()


if __name__ == '__main__':
    main()
