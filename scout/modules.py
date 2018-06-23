from copy import deepcopy


class Module:
    default_params = {}

    def __init__(self, sequencer, name=None, params=None):
        self.sequencer = sequencer
        self._name = name
        self.params = self.merge_params_with_default(params)

    def merge_params_with_default(self, params):
        full_params = deepcopy(self.default_params)
        if params:
            full_params.update(params)
        return full_params

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
