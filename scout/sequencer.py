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
        module = module_class(
            sequencer=sequencer,
            params=config.get("params", {}),
            patches=config.get("patches", []),
        )
        connections = [
            (c["source"]["name"], module.name) for c in config.get("patches", {})
        ]
        sequencer.register(module, connections=connections)


@attr.s
class Sequencer:
    length = attr.ib(default=0)
    modules = attr.ib(default=dict())

    def register(self, module, connections=None):
        if module.name in self.modules:
            raise ValueError(f"Module name {module.name} has already been registered")
        self.modules[module.name] = module
        connections = connections if connections else {}
        for connection in connections:
            self.connect(*connection)

    def connect(self, upstream, downstream):
        pass

    def lookup_value(self, module_name, port):
        return self.modules[module_name].output[port]
