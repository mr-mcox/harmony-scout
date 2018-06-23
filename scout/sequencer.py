import attr
from scout import modules


def build_modules(configs, sequencer):
    for config in configs:
        module_type = config["type"]
        module = None
        if module_type == "rhythm":
            module = modules.Rhythm()
        elif module_type == "sequencer":
            module = modules.Sequencer()
        elif module_type == "consonances":
            module = modules.Consonances()
        else:
            raise ValueError(f"Module type {module_type} unknown")
        sequencer.register(module)


@attr.s
class Sequencer:
    length = attr.ib()
