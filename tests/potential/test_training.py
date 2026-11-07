"""CPU tests of training bookkeeping; no JAX/model weights/simulator required."""

from itertools import chain
from itertools import islice
import json

import numpy as np
import pytest
import safetensors.torch
import torch

from openpi.shared.potential_checkpoint import METADATA_FILENAME
from openpi.shared.potential_checkpoint import validate_checkpoint_metadata
from openpi.training.potential import ResumableBatchSampler
from openpi.training.potential import capture_rng_state
from openpi.training.potential import cost_forecast
from openpi.training.potential import learning_rate
from openpi.training.potential import load_training_checkpoint
from openpi.training.potential import make_ema
from openpi.training.potential import prune_checkpoints
from openpi.training.potential import restore_rng_state
from openpi.training.potential import save_checkpoint
from openpi.training.potential import seed_all
from openpi.training.potential import sha256_file
from openpi.training.potential import stable_seed
from openpi.training.potential import torch_rng_scope
from openpi.training.potential import validate_base_conversion
from openpi.training.potential import validate_data_manifest


def test_sampler_resume_uses_consumed_cursor_not_prefetched_cursor():
    sampler = ResumableBatchSampler(23, 2, 8, 42)
    iterator = iter(sampler)
    observed = list(islice(iterator, 30))  # Includes data a worker could have prefetched.
    resumed = list(islice(iter(ResumableBatchSampler(23, 2, 8, 42, start_microbatch=5)), 25))
    assert resumed == observed[5:]
    # Exactly two effective batches per epoch, with no duplicate examples.
    first_epoch = list(chain.from_iterable(observed[:8]))
    assert len(first_epoch) == len(set(first_epoch)) == 16


def test_sampler_preserves_effective_examples_across_microbatch_sizes():
    small = list(chain.from_iterable(islice(iter(ResumableBatchSampler(23, 2, 8, 42)), 16)))
    large = list(chain.from_iterable(islice(iter(ResumableBatchSampler(23, 8, 8, 42)), 4)))
    assert small == large


def test_accumulation_matches_full_batch_parameter_update():
    seed_all(4)
    full = torch.nn.Linear(3, 2).double()
    accumulated = torch.nn.Linear(3, 2).double()
    accumulated.load_state_dict(full.state_dict())
    inputs, targets = torch.randn(8, 3).double(), torch.randn(8, 2).double()
    full_opt = torch.optim.AdamW(full.parameters(), lr=1e-3)
    micro_opt = torch.optim.AdamW(accumulated.parameters(), lr=1e-3)
    (full(inputs) - targets).square().mean().backward()
    for x, y in zip(inputs.chunk(4), targets.chunk(4), strict=True):
        ((accumulated(x) - y).square().mean() / 4).backward()
    torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(accumulated.parameters(), 1.0)
    full_opt.step()
    micro_opt.step()
    for expected, actual in zip(full.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-12)


def test_ema_blends_first_update_with_theta_zero():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)
    ema = make_ema(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(4)
    ema.update_parameters(model)
    torch.testing.assert_close(ema.module.weight, torch.tensor([[2.2]]))
    assert not ema.module.weight.requires_grad


def test_official_schedule_boundaries():
    schedule = {"warmup_steps": 10_000, "peak_lr": 5e-5, "decay_steps": 1_000_000, "decay_lr": 5e-5}
    assert learning_rate(0, **schedule) == pytest.approx(5e-5 / 10_001)
    assert learning_rate(10_000, **schedule) == pytest.approx(5e-5)
    assert learning_rate(29_999, **schedule) == pytest.approx(5e-5)


def test_rng_streams_do_not_advance_global_rng():
    seed_all(42)
    before = capture_rng_state()
    with torch_rng_scope(stable_seed(42, "corruption", 1, 0), torch.device("cpu")):
        first = torch.randn(5)
    with torch_rng_scope(stable_seed(42, "augmentation", 1, 0), torch.device("cpu")):
        torch.randn(50)
    with torch_rng_scope(stable_seed(42, "corruption", 1, 0), torch.device("cpu")):
        second = torch.randn(5)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(torch.get_rng_state(), before["torch"])
    expected_torch, expected_numpy = torch.rand(3), np.random.random(3)
    restore_rng_state(before)
    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)


def test_metadata_rejects_wrong_or_missing_mode(tmp_path):
    expected = {"velocity_mode": "potential_dot", "action_horizon": 10, "action_dim": 32}
    with pytest.raises(ValueError, match="requires"):
        validate_checkpoint_metadata(tmp_path, **expected)
    assert validate_checkpoint_metadata(tmp_path, **{**expected, "velocity_mode": "direct"}) is None
    (tmp_path / METADATA_FILENAME).write_text(json.dumps({"schema_version": 1, **expected}))
    validate_checkpoint_metadata(tmp_path, **expected)
    with pytest.raises(ValueError, match="velocity_mode"):
        validate_checkpoint_metadata(tmp_path, **{**expected, "velocity_mode": "direct"})
    with pytest.raises(ValueError, match="action_horizon"):
        validate_checkpoint_metadata(tmp_path, **{**expected, "action_horizon": 50})


def test_missing_sidecar_cannot_downgrade_marked_potential(tmp_path):
    (tmp_path / "config.json").write_text('{"velocity_mode": "potential_dot"}')
    with pytest.raises(ValueError, match="evidence"):
        validate_checkpoint_metadata(tmp_path, velocity_mode="direct", action_horizon=10, action_dim=32)


def test_embedded_metadata_prevents_sidecar_removal_and_mode_tampering(tmp_path):
    expected = {"velocity_mode": "potential_dot", "action_horizon": 10, "action_dim": 32}
    metadata = {"schema_version": 1, "trainer": "potential_pi", **expected}
    safetensors.torch.save_file(
        {"weight": torch.ones(1)},
        tmp_path / "model.safetensors",
        metadata={key: str(value) for key, value in metadata.items()},
    )
    with pytest.raises(ValueError, match="requires"):
        validate_checkpoint_metadata(tmp_path, **{**expected, "velocity_mode": "direct"})
    sidecar = tmp_path / METADATA_FILENAME
    sidecar.write_text(json.dumps(metadata))
    validate_checkpoint_metadata(tmp_path, **expected)
    sidecar.write_text(json.dumps({**metadata, "velocity_mode": "direct"}))
    with pytest.raises(ValueError, match="Safetensors velocity_mode"):
        validate_checkpoint_metadata(tmp_path, **{**expected, "velocity_mode": "direct"})
    sidecar.write_text(json.dumps(metadata))
    (tmp_path / "config.json").write_text('{"velocity_mode": "direct"}')
    with pytest.raises(ValueError, match="config.json"):
        validate_checkpoint_metadata(tmp_path, **expected)


def test_research_sidecar_requires_embedded_architecture_metadata(tmp_path):
    expected = {"velocity_mode": "direct", "action_horizon": 10, "action_dim": 32}
    safetensors.torch.save_file({"weight": torch.ones(1)}, tmp_path / "model.safetensors")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps({"schema_version": 1, "trainer": "potential_pi", **expected}))
    with pytest.raises(ValueError, match="missing embedded"):
        validate_checkpoint_metadata(tmp_path, **expected)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.backbone.requires_grad_(requires_grad=False)
        self.expert = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.expert(self.backbone(x))


def _step(model, optimizer, ema):
    optimizer.zero_grad(set_to_none=True)
    x, y = torch.randn(4, 2), torch.randn(4, 1)
    (model(x) - y).square().mean().backward()
    optimizer.step()
    ema.update_parameters(model.expert)


def test_checkpoint_ema_export_and_exact_next_step_resume(tmp_path):
    seed_all(42)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.expert.parameters(), lr=0.01)
    ema = make_ema(model.expert)
    _step(model, optimizer, ema)
    raw = {key: value.clone() for key, value in model.state_dict().items()}
    metadata = {"velocity_mode": "direct", "action_horizon": 10, "action_dim": 32}
    path = save_checkpoint(
        tmp_path / "1",
        model=model,
        expert=model.expert,
        optimizer=optimizer,
        ema=ema,
        state={"optimizer_updates": 1, "consumed_microbatches": 4},
        metadata=metadata,
        assets={"assets/test/norm_stats.json": "{}"},
    )
    # Saving EMA must not alter the live raw training weights.
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, raw[key], atol=0, rtol=0)
    saved_rng = capture_rng_state()
    exported = TinyModel()
    safetensors.torch.load_model(exported, path / "model.safetensors", strict=True)
    torch.testing.assert_close(exported.expert.weight, ema.module.weight)
    torch.testing.assert_close(exported.backbone.weight, model.backbone.weight)
    restore_rng_state(saved_rng)  # Inspection model construction must not change the next training batch.
    _step(model, optimizer, ema)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    expected_ema = ema.module.weight.clone()
    resumed = TinyModel()
    resumed_opt = torch.optim.AdamW(resumed.expert.parameters(), lr=0.01)
    resumed_ema = make_ema(resumed.expert)
    state = load_training_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_opt,
        ema=resumed_ema,
        **metadata,
    )
    assert state["optimizer_updates"] == 1
    _step(resumed, resumed_opt, resumed_ema)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    torch.testing.assert_close(resumed_ema.module.weight, expected_ema, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_checkpoint(
            path,
            model=model,
            expert=model.expert,
            optimizer=optimizer,
            ema=ema,
            state={"optimizer_updates": 1},
            metadata=metadata,
        )


def test_checkpoint_failure_is_not_published_and_restores_weights(tmp_path, monkeypatch):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.expert.parameters())
    ema = make_ema(model.expert)
    with torch.no_grad():
        model.expert.weight.add_(1)
    original = model.expert.weight.clone()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(
            tmp_path / "1",
            model=model,
            expert=model.expert,
            optimizer=optimizer,
            ema=ema,
            state={"optimizer_updates": 1},
            metadata={"velocity_mode": "direct", "action_horizon": 10, "action_dim": 32},
        )
    assert not (tmp_path / "1").exists()
    assert not list(tmp_path.iterdir())
    torch.testing.assert_close(model.expert.weight, original, atol=0, rtol=0)


def test_cost_forecast_counts_accumulated_optimizer_updates():
    forecast = cost_forecast(10, 30_000, 1.8, runs=3)
    assert forecast["gpu_hours"] == pytest.approx(250)
    assert forecast["gpu_cost_usd"] == pytest.approx(450)
    assert forecast["cost_with_20pct_headroom_usd"] == pytest.approx(540)
    no_price = cost_forecast(10, 30_000, None)
    assert no_price["gpu_hours"] == pytest.approx(300_000 / 3600)
    assert no_price["gpu_cost_usd"] is None


def test_dataset_manifest_verifies_revision_coverage_and_bytes(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text("{}")
    (root / "data/episode.parquet").write_bytes(b"test dataset")
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    manifest = {
        "schema_version": 1,
        "repo_id": "physical-intelligence/libero",
        "revision": "a" * 40,
        "dataset_root": str(root),
        "files": files,
    }
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    validate_data_manifest(target, repo_id=manifest["repo_id"], dataset_root=root)
    (root / "data/episode.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_data_manifest(target, repo_id=manifest["repo_id"], dataset_root=root)
    manifest["revision"] = "main"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable"):
        validate_data_manifest(target, repo_id=manifest["repo_id"], dataset_root=root)


def test_retention_only_removes_owned_checkpoints_and_preserves_final(tmp_path):
    for step in (5000, 10000, 30000, 35000, 40000):
        directory = tmp_path / str(step)
        directory.mkdir()
        (directory / METADATA_FILENAME).write_text(json.dumps({"trainer": "potential_pi", "optimizer_updates": step}))
    unowned = tmp_path / "2"
    unowned.mkdir()
    (unowned / METADATA_FILENAME).write_text("{incomplete")
    unrelated = tmp_path / "3"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("unrelated")
    removed = prune_checkpoints(tmp_path, keep_last=2, preserve=(30000,))
    assert {item.name for item in removed} == {"5000", "10000"}
    assert {item.name for item in tmp_path.iterdir()} == {"2", "3", "30000", "35000", "40000"}


def test_base_conversion_requires_fp32_bytes_and_canonical_receipt(tmp_path):
    safetensors.torch.save_file({"weight": torch.ones(1)}, tmp_path / "model.safetensors")
    assert validate_base_conversion(tmp_path, require_receipt=False) is None
    with pytest.raises(ValueError, match="receipt"):
        validate_base_conversion(tmp_path, require_receipt=True)
    receipt = {
        "precision": "float32",
        "source_checkpoint": "/checkpoints/pi05_base/params",
        "source_files_sha256": {"manifest.ocdbt": "a" * 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(receipt))
    assert validate_base_conversion(tmp_path, require_receipt=True) == receipt
    safetensors.torch.save_file({"weight": torch.ones(1).bfloat16()}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="not all FP32"):
        validate_base_conversion(tmp_path, require_receipt=True)
    (tmp_path / "config.json").write_text(json.dumps({**receipt, "precision": "bfloat16"}))
    with pytest.raises(ValueError, match="casting rounded"):
        validate_base_conversion(tmp_path, require_receipt=False)
