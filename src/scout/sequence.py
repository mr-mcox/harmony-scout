import attr


@attr.s
class SequenceInstance:

    chords = attr.ib(default=list())


@attr.s
class SequenceTemplate:

    chord_templates = attr.ib(default=list())

    def generate(self):
        chords = list()
        for i, c in enumerate(self.chord_templates):
            if i > 0:
                chords.append(c.generate([chords[i - 1]]))
            else:
                chords.append(c.generate())
        return SequenceInstance(chords=chords)