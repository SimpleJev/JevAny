"""Recipe overrides and custom adapters report configuration errors clearly."""
import json

import pytest

from jevany.backbones import BackboneAdapter, get_backbone_adapter
from jevany.train import parse_args
from jevany.training import TrainingArgumentParser


@pytest.mark.parametrize("setting", ["rlcr", "multimodal", "eval_before_start"])
def test_boolean_recipe_settings_can_be_disabled_from_the_cli(tmp_path, setting):
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(f'{setting} = true\neval_suite = "evaluation"\n')
    args = ["--config", str(recipe), "--data", "records.jsonl", "--out", str(tmp_path / "run")]
    assert getattr(parse_args(args), setting) is True
    for spelling in (setting, setting.replace("_", "-")):
        assert getattr(parse_args([*args, f"--no-{spelling}"]), setting) is False
        assert getattr(parse_args([*args, f"--no-{spelling}", f"--{spelling}"]), setting) is True


@pytest.mark.parametrize("value", ['"true"', "1"])
def test_boolean_recipe_values_still_require_booleans(tmp_path, value):
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(f"rlcr = {value}\n")
    with pytest.raises(ValueError, match="rlcr must be bool"):
        parse_args(["--config", str(recipe)], parser_class=TrainingArgumentParser)


def test_negative_flags_do_not_change_the_objective_when_resuming(tmp_path):
    saved = vars(parse_args(["--data", "records.jsonl", "--out", str(tmp_path / "original"), "--rlcr"]))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"version": 1, "args": saved}))
    args = ["--resume", str(checkpoint), "--out", str(tmp_path / "continued")]
    assert parse_args(args).rlcr is True
    with pytest.raises(ValueError, match="original training settings: rlcr"):
        parse_args([*args, "--no-rlcr"], parser_class=TrainingArgumentParser)


@pytest.mark.parametrize("name", ["no_such_jevany_adapter:Custom", "jevany.backbones:NoSuchAdapter"])
def test_missing_custom_adapters_identify_the_import_specification(name):
    with pytest.raises(ValueError, match=name) as caught:
        get_backbone_adapter(name)
    assert isinstance(caught.value.__cause__, (ImportError, AttributeError))


def test_custom_adapter_imports_still_require_a_backbone_subclass():
    assert isinstance(get_backbone_adapter("jevany.backbones:BackboneAdapter"), BackboneAdapter)
    with pytest.raises(ValueError, match="subclass"):
        get_backbone_adapter("builtins:object")
