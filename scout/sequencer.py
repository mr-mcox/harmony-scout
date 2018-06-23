import attr
from scout import modules


def build_modules(configs, sequencer):
    for config in configs:
        module_type = config["type"]
        module_class = None
        if module_type == "rhythm":
            module_class = modules.Rhythm
        elif module_type == "sequencer":
            module_class = modules.Sequencer
        elif module_type == "consonances":
            module_class = modules.Consonances
        else:
            raise ValueError(f"Module type {module_type} unknown")
        sequencer.register(module_class(sequencer=sequencer))


@attr.s
class Sequencer:
    length = attr.ib()
