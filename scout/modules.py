from copy import deepcopy
import numpy as np
from collections import namedtuple
from scout.population import normalize_pitch_class

PatchConfig = namedtuple("PatchConfig", ["module", "port", "multiplier"])


class InputLookup:
    def __init__(self, module, patch_config):
        self.sequencer = module.sequencer
        self.default_inputs = {
            item: module.params[item] for item in module.input_parameters
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

    def __setitem__(self, key, value):
        self.default_inputs[key] = value


class Module:
    default_params = dict()
    default_output = dict()
    level = ""
    input_parameters = list()

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
        if self.input["trigger"] != 1:
            return
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

    def evaluate(self, phenotype):
        return 0


class Rhythm(Module):
    """Generate triggers at specified durations

    Parameters
    ----------
    durations: list of int
        Duration of each attack

    Returns
    -------
    out: float
        1 on trigger, 0 otherwise
    """

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


class Seq(Module):
    """Advance sequence on trigger

    Parameters
    ----------
    states: list of float
        Outputs to cycle through

    Returns
    -------
    out: float
        Current state

    """

    default_params = {"states": [0], "trigger": 0}
    input_parameters = ["trigger"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.i = 0
        self.output["out"] = self.params["states"][0]

    def update_outputs(self):
        states = self.params["states"]
        if self.input["trigger"] == 1:
            out = states[self.i % len(states)]
            self.i += 1
            self.output["out"] = out


class Consonances(Judge):
    level = "pitch_class"
    default_params = {"class_value": list(), "trigger": 0}
    input_parameters = ["trigger"]

    def legal_pitch_vals(self):
        class_vals = self.params["class_value"]
        legal_cv = list()
        for c, v in class_vals:
            legal_cv.append((c, v))
            c2 = c - 12
            while c2 >= -12:
                legal_cv.append((c2, v))
                c2 += -12
            c2 = c + 12
            while c2 <= 12:
                legal_cv.append((c2, v))
                c2 += 12
        return legal_cv

    def array_vals(self):
        class_vals = self.legal_pitch_vals()
        return {
            "pitches": [c[0] for c in class_vals],
            "values": [c[1] for c in class_vals],
            "weight": self.input["weight"],
        }

    def evaluate(self, pheno):
        pitches = self.array_output["pitches"]
        values = self.array_output["values"]
        weight = self.array_output["weight"]

        score = 0
        for i in range(pitches.shape[1]):
            matches = pheno == pitches[:, i]
            score += (matches * values[:, i] * weight).sum()
        return score


class CadenceDetector(Judge):
    level = "pitch_class"
    default_params = {
        "weight_authentic": 0,
        "weight_plagel": 0,
        "weight_deceptive": 0,
        "weight_non_authentic": 0,
        "weight_ascending": 0,
        "trigger": 0,
    }
    input_parameters = [
        "trigger",
        "weight_authentic",
        "weight_plagel",
        "weight_deceptive",
        "weight_non_authentic",
        "weight_ascending",
    ]

    def __init__(self, scale, n=3, **kwargs):
        self.scale = scale
        self.n = n
        self.role_num, self.role_ideals = self.construct_ideals()
        super().__init__(**kwargs)

    def scale_steps(self):
        n = self.n
        if n < 3:
            raise ValueError("Functional roles not defined for < 3 note chords")
        prev = {(0, 2, 4)}
        current = prev
        n_notes = 3
        while n - n_notes > 0:
            current = set()
            for base in prev:
                for note in base:
                    new = base + (note,)
                    current.add(new)
                new = base + (len(base) * 2,)  # #adds next triad
                current.add(new)
            n_notes += 1
        return current

    def construct_ideals(self):
        scale = self.scale
        chords = list()
        step_list = self.scale_steps()
        roles = list()
        for role in range(7):
            for steps in step_list:
                roles.append(role)
                chord = list()
                for step in steps:
                    scale_steps = (role + step) % 7
                    chord.append(scale[scale_steps])
                chords.append(chord)
        return np.array(roles), normalize_pitch_class(np.array(chords))

    def functional_role(self, pitch_classes):
        ideals_repeat = np.expand_dims(self.role_ideals, axis=1).repeat(
            pitch_classes.shape[0], axis=1
        )
        diff = np.square(pitch_classes - ideals_repeat).sum(axis=2)
        role_idx = np.argmin(diff, axis=0)
        role_num = self.role_num[role_idx]
        ideals = self.role_ideals[role_idx]
        normalized_pc = pitch_classes / np.linalg.norm(
            pitch_classes, axis=1, keepdims=True
        )
        normalized_ideals = ideals / np.linalg.norm(ideals, axis=1, keepdims=True)
        strengths = np.dot(normalized_pc, normalized_ideals.T)
        return np.array(role_num), np.diag(strengths)

    def array_vals(self):
        return {
            "weight_authentic": self.input["weight_authentic"],
            "weight_plagel": self.input["weight_plagel"],
            "weight_deceptive": self.input["weight_deceptive"],
            "weight_non_authentic": self.input["weight_non_authentic"],
            "weight_ascending": self.input["weight_ascending"],
        }

    def evaluate(self, phenotype):
        weight_authentic = self.array_output["weight_authentic"]
        weight_plagel = self.array_output["weight_plagel"]
        weight_deceptive = self.array_output["weight_deceptive"]
        weight_non_authentic = self.array_output["weight_non_authentic"]
        weight_ascending = self.array_output["weight_ascending"]
        role, strength = self.functional_role(phenotype)
        looped_role = np.concatenate([role, np.expand_dims(role[0], axis=1)], axis=0)
        score = 0
        cadences = {
            "authentic": {(4, 0), (6, 0)},
            "plagel": {(3, 0), (1, 0)},
            "deceptive": {(4, 3), (4, 5)},
        }
        cadences['non_authentic'] = cadences['plagel'] | cadences['deceptive']
        ascending_cadences = self.generate_ascending()
        cadences['ascending'] = ascending_cadences

        for i in range(phenotype.shape[0]):
            movement = (looped_role[i], looped_role[i + 1])
            if movement in cadences["authentic"]:
                score += weight_authentic[i]
            if movement in cadences["plagel"]:
                score += weight_plagel[i]
            if movement in cadences["deceptive"]:
                score += weight_deceptive[i]
            if movement in cadences["non_authentic"]:
                score += weight_non_authentic[i]
            if movement in cadences["ascending"]:
                score += weight_ascending[i]
        return score

    @staticmethod
    def generate_ascending():
        functions = [0, 5, 3, 1, 6, 4]
        movements = set()
        for i, n in enumerate(functions):
            j = 1
            while j < len(functions):
                m = functions[j]
                movements.add((n, m))
                j += 1
        return movements
