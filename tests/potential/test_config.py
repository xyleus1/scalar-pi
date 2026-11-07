import dataclasses
import json

import jax
import pytest
import safetensors.torch
import torch

from openpi.models.pi0_config import Pi0Config
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
