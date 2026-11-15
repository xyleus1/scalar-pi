import dataclasses
import json

import jax
import pytest
import safetensors.torch
import torch

from openpi.models.pi0_config import Pi0Config
from openpi.shared.potential_checkpoint import RESEARCH_PRECISION
from openpi.training.config import get_config


@pytest.mark.parametrize(
    ("name", "mode", "initialization"),
    [
        ("pi05_libero_b0", "direct", "pretrained_output"),
        ("pi05_libero_b1", "direct", "zero_output"),
        ("pi05_libero_potential", "potential_dot", "zero_output"),
    ],
)
def test_protocol_inherits_official_recipe(name, mode, initialization):
    reference = get_config("pi05_libero")
    config = get_config(name)
    assert config.data == reference.data
    assert config.lr_schedule == reference.lr_schedule
    assert config.optimizer == reference.optimizer
    assert config.batch_size == reference.batch_size == 256
    assert config.num_train_steps == reference.num_train_steps == 30_000
    assert config.ema_decay == reference.ema_decay == 0.999
    assert config.model.action_horizon == reference.model.action_horizon == 10
    assert config.model.action_dim == reference.model.action_dim == 32
    assert config.model.velocity_mode == mode
    assert config.model.initialization == initialization
    assert config.model.freeze_vlm
    assert config.model.dtype == "float32"
    assert config.model.pytorch_compile_mode is None
    with pytest.raises(ValueError, match="PyTorch"):
        config.model.create(jax.random.key(0))


def test_potential_configuration_rejects_unsafe_execution_modes():
    config = get_config("pi05_libero_potential").model
    for change in ({"freeze_vlm": False}, {"dtype": "bfloat16"}, {"pytorch_compile_mode": "default"}):
        with pytest.raises(ValueError, match="require"):
            dataclasses.replace(config, **change)
    assert Pi0Config().velocity_mode == "direct"


@pytest.mark.parametrize(
    ("config_name", "message"),
    [("pi05_libero", "expert-only"), ("pi05_libero_b1", "initialization")],
)
def test_loader_rejects_wrong_research_control_before_allocating_model(tmp_path, config_name, message):
    metadata = {
        **RESEARCH_PRECISION,
        "schema_version": 1,
        "trainer": "potential_pi",
        "velocity_mode": "direct",
        "action_horizon": 10,
        "action_dim": 32,
        "initialization": "pretrained_output",
    }
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file({"fixture": torch.ones(1)}, path, metadata={k: str(v) for k, v in metadata.items()})
    (tmp_path / "potential_metadata.json").write_text(json.dumps(metadata))
    config = get_config(config_name)
    with pytest.raises(ValueError, match=message):
        config.model.load_pytorch(config, str(path))


@pytest.mark.parametrize("missing", [False, True])
def test_serving_rejects_old_research_precision_before_model_allocation(tmp_path, monkeypatch, missing):
    from openpi.policies import policy_config

    metadata = {
        **RESEARCH_PRECISION,
        "schema_version": 1,
        "trainer": "potential_pi",
        "velocity_mode": "potential_dot",
        "action_horizon": 10,
        "action_dim": 32,
    }
    if missing:
        del metadata["precision_protocol"]
    else:
        metadata["vlm_precision"] = "bfloat16"
    safetensors.torch.save_file(
        {"fixture": torch.ones(1)},
        tmp_path / "model.safetensors",
        metadata={key: str(value) for key, value in metadata.items()},
    )
    (tmp_path / "potential_metadata.json").write_text(json.dumps(metadata))
    tokenizer = tmp_path / "big_vision/paligemma_tokenizer.model"
    tokenizer.parent.mkdir()
    tokenizer.write_bytes(b"not used before precision rejection")
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda path: tmp_path)
    config = get_config("pi05_libero_potential")
    monkeypatch.setattr(type(config.model), "load_pytorch", lambda *args: pytest.fail("Allocated an unqualified model"))
    with pytest.raises(ValueError, match="precision"):
        policy_config.create_trained_policy(config, tmp_path, pytorch_device="cpu")
