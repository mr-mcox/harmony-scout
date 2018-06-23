import attr
import pytest
from scout.sequencer import build_modules
from scout import modules


@attr.s
class MockSeq:
    modules = attr.ib(default=dict())

    def register(self, module):
        self.modules[module.name] = module

    def connect(self, up, down):
        pass


@pytest.mark.parametrize(
    "module_type,expected_class",
    [
        ("rhythm", modules.Rhythm),
        ("sequencer", modules.Sequencer),
        ("consonances", modules.Consonances),
    ],
)
def test_expected_type(module_type, expected_class):
    config = [{"type": module_type}]
    s = MockSeq()
    build_modules(configs=config, sequencer=s)
    assert isinstance(s.modules[module_type], expected_class)
