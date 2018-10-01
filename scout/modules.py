from copy import deepcopy
import numpy as np
from collections import namedtuple
from scout.population import normalize_pitch_class

PatchConfig = namedtuple("PatchConfig", ["module", "port", "multiplier"])


class InputLookup:
    def __init__(self, module, patch_config):
        self.sequencer = module.sequencer
        self.default_inputs = {
            item: getattr(module, item) for item in module.input_parameters
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
    input_parameters = list()

    def __init__(self, name=None, patches=None, level=None):
        patches = patches if patches else list()
        self.sequencer = None
        self._name = name
        self.output = dict()
        self._input = None
        self.patches = patches
        self.connections = [(c["source"]["name"], self.name) for c in patches]
        self.level = level

    def update_outputs(self):
        pass

    def resolve_step(self):
        self.update_outputs()
        for k, v in self.output.items():
            if v < -1 or v > 1:
                raise ValueError(f"{self.name} has output {k} with value {v}")

    @property
    def name(self):
        if self._name is None:
            self._name = type(self).__name__.lower()
        return self._name

    @property
    def input(self):
        if self._input is None:
            self._input = InputLookup(self, self.patches)
        return self._input


class Judge(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.array_output = dict()

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

    def evaluate(self, phenotype):
        return 0


class Rhythm(Module):
    """Determine clock cycle at step

    Parameters
    ----------
    durations: list of int
        Duration of each attack

    Returns
    -------
    out: float
        Clock time at activation
    """

    def __init__(self, durations=None, **kwargs):
        super().__init__(**kwargs)
        self.durations = [1] if durations is None else durations
        self.i = 0
        self.total = 0
        self.next_output = 0
        self.output = {"out": self.next_output}

    def update_outputs(self):
        self.output["out"] = self.next_output
        self.total += self.durations[self.i]
        total_durations = sum(self.durations)
        self.next_output = (self.total % total_durations) / total_durations
        self.i = (self.i + 1) % len(self.durations)


class Seq(Module):
    """Advance sequence

    Parameters
    ----------
    states: list of float
        Outputs to cycle through

    Returns
    -------
    out: float
        Current state

    """

    input_parameters = ["clock"]

    def __init__(self, states=None, **kwargs):
        super().__init__(**kwargs)
        states = [0] if states is None else states
        self.states = states
        self.i = 0
        self.output["out"] = self.states[0]
        self.clock = None

    def update_outputs(self):
        states = self.states
        if self.input["clock"] is None:
            self.output["out"] = states[self.i % len(states)]
            self.i += 1
        else:
            i = int(self.input["clock"] * len(states))
            self.output["out"] = states[i]


class Sawtooth(Module):

    input_parameters = ["clock"]

    def __init__(self, period=4, is_ac=True, phase=0, invert=False, **kwargs):
        super().__init__(**kwargs)
        self.output = {"out": -1}
        self.period = period
        self.i = 0
        self.phase = phase
        self.is_ac = is_ac
        self.invert = invert
        self.clock = None

    def update_outputs(self):
        period = self.period
        clock = self.input["clock"]
        if clock is None:
            clock = self.i
            self.i += 1
        out = ((self.phase + clock) % period) / period
        if self.invert:
            out = 1 - out
        if self.is_ac:
            out = out * 2 - 1
        self.output["out"] = out


class Consonances(Judge):
    def __init__(self, class_value=None, **kwargs):
        super().__init__(level="pitch_class", **kwargs)
        self.class_value = list() if class_value is None else class_value

    def legal_pitch_vals(self):
        class_vals = self.class_value
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
    input_parameters = [
        "weight_authentic",
        "weight_plagel",
        "weight_deceptive",
        "weight_non_authentic",
        "weight_ascending",
    ]

    def __init__(
        self,
        scale,
        n=3,
        weight_authentic=0,
        weight_plagel=0,
        weight_deceptive=0,
        weight_non_authentic=0,
        weight_ascending=0,
        **kwargs,
    ):
        super().__init__(level="pitch_class", **kwargs)
        self.scale = scale
        self.n = n
        self.role_num, self.role_ideals = self.construct_ideals()
        self.weight_authentic = weight_authentic
        self.weight_plagel = weight_plagel
        self.weight_deceptive = weight_deceptive
        self.weight_non_authentic = weight_non_authentic
        self.weight_ascending = weight_ascending

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
        cadences["non_authentic"] = cadences["plagel"] | cadences["deceptive"]
        ascending_cadences = self.generate_ascending()
        cadences["ascending"] = ascending_cadences

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
