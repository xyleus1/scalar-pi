"""Actual tiny PI05 hybrid raw/EMA checkpoints and exact matched-arm restart."""

import copy

import pytest
from safetensors import safe_open
import safetensors.torch
from test_pi05_integration import tiny_model  # noqa: F401 -- shared real production-model fixture.
import torch

from openpi.shared.potential_checkpoint import validate_checkpoint_metadata
from openpi.shared.potential_precision import RESEARCH_PRECISION
from openpi.shared.potential_precision import VLM_PREFIX
from openpi.shared.potential_precision import checkpoint_float_dtype
from openpi.training.potential import ResumableBatchSampler
from openpi.training.potential import load_training_checkpoint
from openpi.training.potential import make_ema
from openpi.training.potential import save_checkpoint
from openpi.training.potential import stable_seed
from openpi.training.potential import torch_rng_scope


def prepare(model, arm):
    torch.set_float32_matmul_precision("highest")
    model.velocity_mode = "potential_dot" if arm == "P" else "direct"
    model.configure_expert_only(vlm_dtype=torch.bfloat16)
    if arm != "B0":
        model.zero_action_output()
    model.validate_research_precision()
    model.train()


def snapshot(model):
    return {name: value.clone() for name, value in model.state_dict().items()}


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        assert value.dtype == right[name].dtype, name
        torch.testing.assert_close(value, right[name], atol=0, rtol=0)


def metadata(model, signature):
    return {
        **RESEARCH_PRECISION,
        "velocity_mode": model.velocity_mode,
        "action_horizon": 3,
        "action_dim": 4,
        "resume_signature": signature,
    }


def update(model, observation, actions, optimizer, ema, *, number, cursor):
    sampler = iter(ResumableBatchSampler(len(actions), 2, 4, 42, start_microbatch=cursor))
    trace = []
    optimizer.zero_grad(set_to_none=True)
    for micro in range(2):
        indices = next(sampler)
        batch = actions[indices]
        with torch_rng_scope(stable_seed(42, "corruption", number, micro), torch.device("cpu")):
            noise = model.sample_noise(batch.shape, "cpu")
            time = model.sample_time(len(batch), "cpu")
        with torch_rng_scope(stable_seed(42, "augmentation", number, micro), torch.device("cpu")):
            loss = model(observation, batch, noise=noise, time=time).mean() / 2
            assert torch.isfinite(loss)
            loss.backward()
        trace.append((indices, noise.clone(), time.clone()))
    torch.nn.utils.clip_grad_norm_(model.expert_modules().parameters(), 1.0)
    optimizer.step()
    ema.update_parameters(model.expert_modules())
    return trace


@pytest.mark.parametrize("arm", ["B0", "B1", "P"])
def test_mixed_native_headers_ema_aliases_and_exact_next_update(tiny_model, tmp_path, arm):  # noqa: F811
    model, observation = tiny_model
    prepare(model, arm)
    initial = copy.deepcopy(model)
    frozen = {name: value for name, value in snapshot(model).items() if name.startswith(VLM_PREFIX)}
    actions = torch.linspace(-1, 1, 8 * 3 * 4).reshape(8, 3, 4)
    optimizer = torch.optim.AdamW(model.expert_modules().parameters(), lr=1e-3)
    ema = make_ema(model.expert_modules())
    signature = {**RESEARCH_PRECISION, "seed": 42, "arm": arm, "source_snapshot_sha256": "a" * 64}
    update(model, observation, actions, optimizer, ema, number=0, cursor=0)
    raw = snapshot(model)
    saved = save_checkpoint(
        tmp_path / "1",
        model=model,
        expert=model.expert_modules(),
        optimizer=optimizer,
        ema=ema,
        state={"optimizer_updates": 1, "consumed_microbatches": 2, "resume_signature": signature},
        metadata=metadata(model, signature),
    )
    assert_state_equal(raw, model.state_dict())
    assert_state_equal(
        frozen, {name: value for name, value in model.state_dict().items() if name.startswith(VLM_PREFIX)}
    )
    actual = validate_checkpoint_metadata(saved, velocity_mode=model.velocity_mode, action_horizon=3, action_dim=4)
    assert {key: actual[key] for key in RESEARCH_PRECISION} == RESEARCH_PRECISION
    state = model.state_dict()
    aliases = {}
    for name, tensor in state.items():
        aliases.setdefault(tensor.data_ptr(), []).append(name)
    tied = [names for names in aliases.values() if len(names) > 1 and any("embed_tokens" in name for name in names)]
    assert tied, "The native tied VLM embedding/head must remain exercised"
    for filename in ("model.safetensors", "raw_model.safetensors"):
        with safe_open(saved / filename, framework="pt", device="cpu") as weights:
            names = set(weights.keys())
            dtypes = set()
            for name in names:
                dtype = weights.get_slice(name).get_dtype()
                if state[name].is_floating_point():
                    assert dtype == checkpoint_float_dtype(name), name
                    dtypes.add(dtype)
            assert dtypes == {"BF16", "F32"}
            for group in tied:
                assert len(names.intersection(group)) == 1
    # Production constructs an FP32 architecture, loads mixed saved tensors
    # into it, and only then reapplies the frozen-VLM cast policy.
    served = copy.deepcopy(initial).float()
    safetensors.torch.load_model(served, saved / "model.safetensors", strict=True)
    served.configure_expert_only()
    served.validate_research_precision()
    assert_state_equal(ema.module.state_dict(), served.expert_modules().state_dict())
    assert_state_equal(
        frozen, {name: value for name, value in served.state_dict().items() if name.startswith(VLM_PREFIX)}
    )
    expected_trace = update(model, observation, actions, optimizer, ema, number=1, cursor=2)
    resumed = initial.float()
    safetensors.torch.load_model(resumed, saved / "raw_model.safetensors", strict=True)
    resumed.configure_expert_only()
    resumed.validate_research_precision()
    assert_state_equal(raw, resumed.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed.expert_modules().parameters(), lr=1e-3)
    resumed_ema = make_ema(resumed.expert_modules())
    restored = load_training_checkpoint(
        saved,
        model=resumed,
        optimizer=resumed_optimizer,
        ema=resumed_ema,
        velocity_mode=model.velocity_mode,
        action_horizon=3,
        action_dim=4,
        expected_resume_signature=signature,
        load_weights=False,
    )
    actual_trace = update(
        resumed,
        observation,
        actions,
        resumed_optimizer,
        resumed_ema,
        number=restored["optimizer_updates"],
        cursor=restored["consumed_microbatches"],
    )
    for expected, actual in zip(expected_trace, actual_trace, strict=True):
        assert expected[0] == actual[0]
        for left, right in zip(expected[1:], actual[1:], strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert_state_equal(model.state_dict(), resumed.state_dict())
    assert_state_equal(ema.state_dict(), resumed_ema.state_dict())
    assert optimizer.state_dict()["param_groups"] == resumed_optimizer.state_dict()["param_groups"]
    for key, values in optimizer.state_dict()["state"].items():
        assert_state_equal(values, resumed_optimizer.state_dict()["state"][key])


def test_failed_mixed_ema_save_restores_raw_live_weights(tiny_model, tmp_path, monkeypatch):  # noqa: F811
    model, observation = tiny_model
    prepare(model, "P")
    optimizer = torch.optim.AdamW(model.expert_modules().parameters(), lr=1e-3)
    ema = make_ema(model.expert_modules())
    actions = torch.linspace(-1, 1, 8 * 3 * 4).reshape(8, 3, 4)
    update(model, observation, actions, optimizer, ema, number=0, cursor=0)
    raw = snapshot(model)
    actual_save = safetensors.torch.save_model

    def fail_ema(network, filename, **kwargs):
        if kwargs["metadata"]["weight_kind"] == "ema":
            assert_state_equal(ema.module.state_dict(), network.expert_modules().state_dict())
            raise RuntimeError("Injected EMA publication failure")
        return actual_save(network, filename, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_model", fail_ema)
    with pytest.raises(RuntimeError, match="Injected EMA publication"):
        save_checkpoint(
            tmp_path / "1",
            model=model,
            expert=model.expert_modules(),
            optimizer=optimizer,
            ema=ema,
            state={"optimizer_updates": 1, "consumed_microbatches": 2},
            metadata=metadata(model, {}),
        )
    assert_state_equal(raw, model.state_dict())
    assert list(tmp_path.iterdir()) == []
