from copy import deepcopy
import numpy as np
from collections import namedtuple

PatchConfig = namedtuple("PatchConfig", ["module", "port", "multiplier"])


class InputLookup:
    def __init__(self, module, patch_config):
        self.sequencer = module.sequencer
        self.default_inputs = {
            k: module.params[v] for k, v in module.input_parameter_map.items()
        }
        patch_lookup = dict()
        patch_config = patch_config if patch_config else list()
        for item in patch_config:
            values = PatchConfig(
                item["source"]["name"],
                item["source"].get("port", "out"),
                item.get("mult", 1),
            )
            patch_lookup[item["dest"]] = values
        self.patch_lookup = patch_lookup

    def __getitem__(self, item):
        if item in self.patch_lookup:
            patch = self.patch_lookup[item]
            return self.sequencer.lookup_value(patch.module, patch.port)
        else:
            return self.default_inputs[item]


class Module:
    default_params = {}
    default_output = {}
    level = ""
    input_parameter_map = {}

    def __init__(self, sequencer, name=None, params=None, patches=None):
        patches = patches if patches else list()
        self.sequencer = sequencer
        self._name = name
        self.params = self.merge_params_with_default(params)
        self.output = deepcopy(self.default_output)
        self.input = InputLookup(self, patches)
        connections = [(c["source"]["name"], self.name) for c in patches]
        sequencer.register(self, connections=connections)

    def merge_params_with_default(self, params):
        full_params = deepcopy(self.default_params)
        if params:
            full_params.update(params)
        return full_params

    def update_outputs(self):
        pass

    def resolve_step(self):
        self.update_outputs()
        for k, v in self.output.items():
            if v < -1 or v > 1:
                raise ValueError(f"{self.name} has output {k} with value {v}")

    @property
    def name(self):
        if not self._name:
            return type(self).__name__.lower()


class Judge(Module):
    def __init__(self, **kwargs):
        self.array_output = dict()
        super().__init__(**kwargs)

    def array_vals(self):
        return dict()

    def resolve_step(self):
        super().resolve_step()
        array_output = self.array_output
        for k, v in self.array_vals().items():
            if k in array_output:
                prev_array = array_output[k]
                new_array = np.array([v])
                array = np.concatenate((prev_array, new_array))
                array_output[k] = array
            else:
                array_output[k] = np.array([v])
        self.array_output = array_output


class Rhythm(Module):
    default_output = {"out": 1}
    default_params = {"durations": [1]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        durations = self.params["durations"]
        activations = list()
        for duration in durations:
            activations.extend([1] + [0] * (duration - 1))
        self.activations = activations
        self.i = 0

    def update_outputs(self):
        out = self.activations[self.i % len(self.activations)]
        self.output["out"] = out
        self.i += 1


class Sequencer(Module):
    pass


class Consonances(Module):
    pass
