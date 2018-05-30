import attr
from copy import deepcopy
import numpy as np


@attr.s
class SequenceInstance:

    chords = attr.ib(default=list())
    scoring_rules = attr.ib(default=list())

    def pitches_as_array(self):
        return np.stack([c.pitches for c in self.chords])

    def score(self):
        total = 0
        for mul, rule in self.scoring_rules:
            total += mul * rule(self)
        return total


def graph_children(graph, parent):
    return {c for p, c in graph if p == parent}


def graph_parents(graph, child):
    return {p for p, c in graph if c == child}


class SequenceTemplate:
    def __init__(self, chord_templates=None, scoring_rules=None):
        self.chord_templates = list(
        ) if chord_templates is None else chord_templates
        self._links = set()
        self.scoring_rules = list() if scoring_rules is None else scoring_rules

    def generate(self):
        chord_by_idx = dict()
        for idx in self.eval_order():
            deps = [chord_by_idx[i] for i in self.get_dependencies(idx)]
            chord_by_idx[idx] = self.chord_templates[idx].generate(deps)
        chords = [chord_by_idx[i] for i in range(len(self.chord_templates))]

        return SequenceInstance(chords=chords, scoring_rules=self.scoring_rules)

    def add_link(self, parent, child):
        self._links.add((parent, child))

    def get_dependencies(self, child):
        deps = set()
        for p, c in self._links:
            if c == child:
                deps.add(p)
        return deps

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
            ValueError('Values remained on graph. It cannot be a DAG')
        return order
