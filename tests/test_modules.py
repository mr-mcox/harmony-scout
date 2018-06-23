from scout.modules import Module
import attr


@attr.s
class MockSeq:
    modules = attr.ib(default=dict())

    def register(self, module):
        self.modules[module.name] = module

    def connect(self, up, down):
        pass


class MyModule(Module):
    default_params = {"default": "default", "override": "old"}


def test_default_params():
    m = MyModule(sequencer=MockSeq())
    assert m.params["default"] == "default"


def test_update_params():
    m = MyModule(sequencer=MockSeq(), params={"override": "new"})
    assert m.params["override"] == "new"
