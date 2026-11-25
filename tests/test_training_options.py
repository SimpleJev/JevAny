"""Recipe overrides and custom adapters report configuration errors clearly."""
import json

import pytest

from jevany.backbones import BackboneAdapter, get_backbone_adapter
from jevany.train import parse_args
from jevany.training import TrainingArgumentParser


@pytest.mark.parametrize("setting,value,message", [
    ("head_dim", 0, "head dimension must be positive"),
    ("head_dim", -1, "head dimension must be positive"),
    ("replay", -1, "replay must be nonnegative"),
    ("seed", -1, "seed must be in"),
    ("seed", 2**32, "seed must be in"),
])
def test_invalid_training_limits_are_rejected_before_loading_inputs(tmp_path, monkeypatch, setting, value, message):
    from jevany.train import main

    def unexpected_preflight(*args):
        pytest.fail("invalid options must be rejected before reading data or loading weights")

    monkeypatch.setattr("jevany.train.preflight", unexpected_preflight)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match=message):
        main([
            "--data", "records.jsonl", "--suite", "suite", "--out", str(output),
            f"--{setting.replace('_', '-')}", str(value),
        ], parser_class=TrainingArgumentParser)
    assert not output.exists()


@pytest.mark.parametrize("setting,value", [
    ("head_dim", 1), ("replay", 0), ("replay", 1), ("seed", 0), ("seed", 2**32 - 1),
])
def test_valid_training_limit_boundaries(tmp_path, setting, value):
    parsed = parse_args([
        "--data", "records.jsonl", "--suite", "suite", "--out", str(tmp_path / "run"),
        f"--{setting.replace('_', '-')}", str(value),
    ])
    assert getattr(parsed, setting) == value


@pytest.mark.parametrize("entrypoint", ["cli", "python"])
@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "broken-symlink"])
def test_occupied_training_outputs_fail_before_inputs(tmp_path, monkeypatch, capsys, entrypoint, kind):
    from jevany.train import main
    from jevany.training import train

    output = tmp_path / "run"
    retained = tmp_path / "retained"
    if kind == "directory":
        output.mkdir()
        retained = output / "keep.txt"
        retained.write_text("keep")
    elif kind == "file":
        output.write_text("keep")
        retained = output
    else:
        if kind == "symlink":
            retained.mkdir()
            (retained / "keep.txt").write_text("keep")
        output.symlink_to(retained, target_is_directory=True)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr("jevany.train.preflight",
                        lambda *args: pytest.fail("occupied outputs must fail before reading data or loading weights"))
    if entrypoint == "cli":
        with pytest.raises(SystemExit) as error:
            main(["--data", "records.jsonl", "--out", str(output)])
        assert error.value.code == 2
        message = capsys.readouterr().err
    else:
        recipe = tmp_path / "recipe.toml"
        recipe.write_text('data = "records.jsonl"\n')
        with pytest.raises(ValueError) as error:
            train(recipe, output_dir=output)
        message = str(error.value)
    assert "refusing to overwrite an existing run" in message
    assert str(output) in message
    if kind in ("directory", "file"):
        assert retained.read_text() == "keep"
    else:
        assert output.is_symlink() and output.readlink() == retained
        if kind == "symlink":
            assert (retained / "keep.txt").read_text() == "keep"
        else:
            assert not retained.exists()


def test_dry_run_rejects_a_broken_output_symlink(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.symlink_to(tmp_path / "missing", target_is_directory=True)
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="refusing to overwrite an existing run"):
        parse_args(["--data", "records.jsonl", "--out", str(output), "--dry-run"],
                   parser_class=TrainingArgumentParser)


def test_training_accepts_a_new_output_under_a_directory_symlink(tmp_path, monkeypatch):
    directory = tmp_path / "runs"
    directory.mkdir()
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    output = link / "new-run"
    monkeypatch.setenv("RANK", "0")
    assert parse_args(["--data", "records.jsonl", "--out", str(output)]).out == str(output)
    assert not (directory / "new-run").exists()


def test_nonzero_rank_accepts_output_already_created_by_rank_zero(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setenv("RANK", "1")
    assert parse_args(["--data", "records.jsonl", "--out", str(output)]).out == str(output)


def test_lm_token_mode_does_not_require_a_pointer_head_dimension(tmp_path):
    parsed = parse_args([
        "--data", "records.jsonl", "--out", str(tmp_path / "run"),
        "--decision-mode", "lm_token", "--head-dim", "0",
    ])
    assert parsed.head_dim == 0


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
