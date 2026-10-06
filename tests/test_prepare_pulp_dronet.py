import importlib.util
from pathlib import Path


def load_prepare_module():
    script = Path(__file__).parents[1] / "tools" / "prepare_pulp_dronet.py"
    spec = importlib.util.spec_from_file_location("prepare_pulp_dronet", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_mapping() -> None:
    module = load_prepare_module()
    assert module.command_for(0.0, 1.0, 0.2, 0.5) == "stop"
    assert module.command_for(0.3, 0.0, 0.2, 0.5) == "left"
    assert module.command_for(-0.3, 0.0, 0.2, 0.5) == "right"
    assert module.command_for(0.2, 0.0, 0.2, 0.5) == "forward"


def test_safe_name_prevents_acquisition_collisions(tmp_path: Path) -> None:
    module = load_prepare_module()
    first = tmp_path / "pilot_a" / "acquisition1"
    second = tmp_path / "pilot_b" / "acquisition1"
    assert module.safe_name(tmp_path, first, "1.jpeg") != module.safe_name(
        tmp_path, second, "1.jpeg"
    )

