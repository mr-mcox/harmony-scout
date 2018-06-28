from scout.population import pitch_classes_with_pitch
from scout.lyndon import standard_lyndon


def main():
    diatonic = [0, 2, 4, 5, 7, 9, 11]
    pcs = pitch_classes_with_pitch(diatonic, n=4)
    chord_shapes = set()
    print(pcs.shape[0])
    # print(pcs)
    for i in range(pcs.shape[0]):
        chord_shapes.add(tuple(standard_lyndon(pcs[i])))
    print(len(chord_shapes))
    # print(chord_shapes)


if __name__ == "__main__":
    main()
