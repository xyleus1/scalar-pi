"""Actual reduced Gemma/PI05 paths; required in the candidate model image."""

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma import modeling_gemma

from openpi.research.policy_checks import array_error
from openpi.research.uncached_reference import reference_inputs
from openpi.research.uncached_reference import torch_joint_velocity
from openpi.research.uncached_reference import trajectory
from openpi.research.uncached_reference import uncached_reference


@pytest.mark.parametrize("masked", [False, True])
def test_fp32_oracle_matches_native_joint_cache_and_all_solver_states(tiny_model, monkeypatch, masked):
    model, observation = tiny_model
    model.velocity_mode = "direct"
    model.eval()
    torch.set_float32_matmul_precision("highest")
    if masked:
        preprocess = model._preprocess_observation  # noqa: SLF001

        def padded(obs, *, train):
            images, masks, tokens, padding, state = preprocess(obs, train=train)
            masks = [item.clone() for item in masks]
            masks[0][1] = False
            padding = padding.clone()
            padding[1, -1] = False
            return images, masks, tokens, padding, state

        monkeypatch.setattr(model, "_preprocess_observation", padded)
    noise, time = torch.randn(2, 3, 4), torch.tensor([0.25, 0.75])
    with torch.no_grad():
        prepared, context = reference_inputs(model, observation), model.encode_context(observation)
        oracle, keys = uncached_reference(model, prepared, noise, time, trace=True)
        native = torch_joint_velocity(model, observation, noise, time)
        cached = model.predict_velocity(noise, time, context)
        torch.testing.assert_close(oracle, native, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(oracle, cached, atol=1e-4, rtol=1e-4)
        for actual, expected in zip(context.past_key_values, keys, strict=True):
            for left, right in zip(actual, expected, strict=True):
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
        for steps in (1, 10):
            ref = trajectory(lambda value, t: uncached_reference(model, prepared, value, t), noise, steps)
            native = trajectory(lambda value, t: torch_joint_velocity(model, observation, value, t), noise, steps)
            assert array_error(ref, native, atol=1e-4, rtol=1e-4)["passed"]
            sampled = model.sample_actions("cpu", observation, noise=noise, num_steps=steps).numpy()
            assert array_error(ref[:, -1], sampled, atol=1e-4, rtol=1e-4)["passed"]


def test_hybrid_oracle_uses_branch_dtypes_without_cache_or_joint_api(tiny_model, monkeypatch):
    model, observation = tiny_model
    model.configure_expert_only()
    model.validate_research_precision()
    model.velocity_mode = "direct"
    model.eval()
    noise, time = torch.randn(2, 3, 4), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        prepared, context = reference_inputs(model, observation), model.encode_context(observation)
        expected = model.predict_velocity(noise, time, context)
        before = [(k.clone(), v.clone()) for k, v in context.past_key_values]

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent oracle must never use a cache or joint forward")

    monkeypatch.setattr(DynamicCache, "__init__", forbidden)
    monkeypatch.setattr(model.paligemma_with_expert, "forward", forbidden)
    attention, dtypes = modeling_gemma.eager_attention_forward, []

    def traced(module, q, k, v, *args, **kwargs):
        dtypes.append((q.dtype, k.dtype, v.dtype))
        return attention(module, q, k, v, *args, **kwargs)

    monkeypatch.setattr(modeling_gemma, "eager_attention_forward", traced)
    with torch.no_grad():
        actual = uncached_reference(model, prepared, noise, time)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert dtypes == [(torch.bfloat16,) * 3, (torch.float32,) * 3] * 2
    for now, old in zip(context.past_key_values, before, strict=True):
        for left, right in zip(now, old, strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_hybrid_corrupted_cache_is_detected_by_independent_oracle(tiny_model):
    model, observation = tiny_model
    model.configure_expert_only()
    model.velocity_mode = "direct"
    model.eval()
    noise, time = torch.randn(2, 3, 4), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        prepared, context = reference_inputs(model, observation), model.encode_context(observation)
        oracle = uncached_reference(model, prepared, noise, time)
        for _, value in context.past_key_values:
            value.add_(4)
        damaged = model.predict_velocity(noise, time, context)
    assert not array_error(damaged.numpy(), oracle.numpy(), atol=1e-4, rtol=1e-4)["passed"]


def test_hybrid_oracle_supports_full_action_double_backward(tiny_model):
    model, observation = tiny_model
    model.configure_expert_only()
    model.eval()
    with torch.no_grad():
        prepared = reference_inputs(model, observation)
    actions = torch.randn(2, 3, 4, requires_grad=True)
    raw = uncached_reference(model, prepared, actions, torch.tensor([0.2, 0.8]))
    potential = (actions * raw).flatten(1).sum(1)
    velocity = torch.autograd.grad(potential.sum(), actions, create_graph=True)[0]
    velocity.square().mean().backward()
    assert torch.isfinite(velocity).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.expert_modules().parameters())
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
