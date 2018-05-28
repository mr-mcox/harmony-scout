import mido
import numpy as np
from time import sleep


def main():
    outport = mido.open_output('IAC Driver Bus 1')

    base_chord = np.array([60, 67, 72])
    ts = [0, 3, 7, 0]
    chords = list()
    for t in ts:
        chords.append(base_chord + t)
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
