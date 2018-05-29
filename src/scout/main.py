import mido
import numpy as np
from scout.spaces import ScaleSpace, PitchSpace
from time import sleep


def main():
    outport = mido.open_output('IAC Driver Bus 1')

    base_chord = np.array([0, 3, 5])
    ts = [0, 2, 5, 0]
    chords = list()
    scale = ScaleSpace(root=60)
    pitch_space = PitchSpace()
    for t in ts:
        chords.append(scale.pitch_values(pitch_space, base_chord + t))
    for chord in chords:
        for note in np.nditer(chord):
            msg = mido.Message('note_on', note=int(note), velocity=100)
            outport.send(msg)
        sleep(1)
        for note in np.nditer(chord):
            outport.send(mido.Message('note_off', note=int(note)))
    outport.close()


if __name__ == '__main__':
    main()
