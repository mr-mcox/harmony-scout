import attr


class Module:
    def __init__(self):
        self._name = None

    @property
    def name(self):
        if not self._name:
            return type(self).__name__.lower()


class Rhythm(Module):
    pass


class Sequencer(Module):
    pass


class Consonances(Module):
    pass
