import pytest
from scout.sequencer import build_modules
from scout import modules
from unittest.mock import patch
from scout.sequencer import Sequencer


@pytest.mark.parametrize(
    "module_type,expected_class",
    [
        ("rhythm", modules.Rhythm),
        ("seq", modules.Seq),
        ("consonances", modules.Consonances),
    ],
)
def test_expected_type(module_type, expected_class):
    config = [{"type": module_type}]
    s = Sequencer()
    build_modules(configs=config, sequencer=s)
    assert isinstance(s.modules[module_type], expected_class)


def test_register_with_connection():
    config = [{"type": "rhythm", "patches": [{"source": {"name": "upstream"}, 'dest':'input'}]}]
    s = Sequencer()
    with patch.object(Sequencer, "connect") as mock_connect:
        build_modules(config, sequencer=s)
    mock_connect.assert_called_with("upstream", "rhythm")
