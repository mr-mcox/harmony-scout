import attr
from scout import modules
from copy import deepcopy
from collections import defaultdict


def build_modules(configs, sequencer):
    for config in configs:
        module_type = config["type"]
        module_class = None
        if module_type == "rhythm":
            module_class = modules.Rhythm
        elif module_type == "seq":
            module_class = modules.Seq
        elif module_type == "consonances":
            module_class = modules.Consonances
        else:
            raise ValueError(f"Module type {module_type} unknown")
        module = module_class(
            sequencer=sequencer,
            params=config.get("params", {}),
            patches=config.get("patches", []),
        )


def graph_children(graph, parent):
    return {c for p, c in graph if p == parent}


def graph_parents(graph, child):
    return {p for p, c in graph if c == child}


class Sequencer:
    def __init__(self, length=0):
        self.length = length
        self.modules = dict()
        self.for_level = defaultdict(list)
        self._links = set()

    def register(self, module, connections=None):
        if module.name in self.modules:
            raise ValueError(f"Module name {module.name} has already been registered")
        self.modules[module.name] = module
        if module.level:
            self.for_level[module.level].append(module)
        connections = connections if connections else {}
        for connection in connections:
            self.connect(*connection)

    def connect(self, parent, child):
        self._links.add((parent, child))

    def lookup_value(self, module_name, port):
        return self.modules[module_name].output[port]

    def eval_order(self):
        # Kahn's algorithm for topological sorting
        graph = deepcopy(self._links)
        no_parent = set()
        order = list()
        for i in range(len(self.chord_templates)):
            if len(graph_parents(graph, i)) == 0:
                no_parent.add(i)
        while len(no_parent) > 0:
            n = no_parent.pop()
            order.append(n)
            for child in graph_children(graph, n):
                graph.remove((n, child))
                if len(graph_parents(graph, child)) == 0:
                    no_parent.add(child)
        if len(graph) > 0:
            ValueError("Values remained on graph. It cannot be a DAG")
        return order

    def sequence(self):
        eval_order = self.eval_order()
        for i in range(self.length):
            self.sequence_step(eval_order)

    def sequence_step(self, eval_order):
        for module_name in eval_order:
            self.modules[module_name].resolve_step()
