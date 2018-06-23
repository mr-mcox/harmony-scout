from scout.modules import Module
from scout.sequencer import Sequencer


class MyModule(Module):
    default_params = {"default": "default", "override": "old", "add_num": 1}
    input_parameter_map = {"add_num": "add_num"}
    default_output = {"out": 0}

    def update_outputs(self):
        self.output["out"] += self.input["add_num"]


class MyInputModule(Module):
    default_output = {"out": 0.5}


def test_default_params():
    m = MyModule(sequencer=Sequencer())
    assert m.params["default"] == "default"


def test_update_params():
    m = MyModule(sequencer=Sequencer(), params={"override": "new"})
    assert m.params["override"] == "new"


def test_values():
    m = MyModule(sequencer=Sequencer())
    assert m.output["out"] == 0


def test_resolve_step():
    m = MyModule(sequencer=Sequencer())
    m.resolve_step()
    assert m.output["out"] == 1


def test_resolve_step_with_input():
    s = Sequencer()
    s.register(MyInputModule(sequencer=s))
    m = MyModule(
        sequencer=s, patches=[{"source": {"name": "myinputmodule"}, "dest": "add_num"}]
    )
    m.resolve_step()
    assert m.output["out"] == 0.5
