from scout.modules import Module
from scout.sequencer import Sequencer


class MyModule(Module):
    input_parameters = ["add_num"]

    def __init__(self, default="default", override="old", add_num=1, **kwargs):
        super().__init__(**kwargs)
        self.default = default
        self.override = override
        self.add_num = add_num
        self.output = {"out": 0}

    def update_outputs(self):
        self.output["out"] += self.input["add_num"]


class MyInputModule(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = {"out": 0.5}


def test_default_params():
    m = MyModule(sequencer=Sequencer())
    assert m.default == "default"


def test_update_params():
    m = MyModule(sequencer=Sequencer(), override="new")
    assert m.override == "new"


def test_values():
    m = MyModule(sequencer=Sequencer())
    assert m.output["out"] == 0


def test_resolve_step():
    m = MyModule(sequencer=Sequencer())
    m.resolve_step()
    assert m.output["out"] == 1


def test_resolve_step_with_input():
    s = Sequencer()
    MyInputModule(sequencer=s)
    m = MyModule(
        sequencer=s, patches=[{"source": {"name": "myinputmodule"}, "dest": "add_num"}]
    )
    m.resolve_step()
    assert m.output["out"] == 0.5
