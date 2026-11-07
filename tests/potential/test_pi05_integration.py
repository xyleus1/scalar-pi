"""Exercise the actual patched Gemma/PI05 paths at reduced dimensions on CPU.

Only model sizes and the image preprocessing fixture are reduced. The expert,
adaptive normalization, cache, potential, and double backward are production code.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers import SiglipVisionConfig

from openpi.models import gemma
from openpi.models_pytorch import gemma_pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


@pytest.fixture
def tiny_model(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(14)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda name: gemma.Config(
            width=32 if name == "vlm" else 16, depth=2, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=4
        ),
    )

    def tiny_vlm(config):
        config.text_config.vocab_size = 128
        config.image_token_index = 127
        config.vision_config = SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=14,
            patch_size=7,
            projection_dim=config.text_config.hidden_size,
            vision_use_head=False,
        )
        return PaliGemmaForConditionalGeneration(config)

    def tiny_expert(config):
        config.vocab_size = 128
        return GemmaForCausalLM(config)

    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", tiny_vlm)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", tiny_expert)
    config = SimpleNamespace(
        pi05=True,
        paligemma_variant="vlm",
        action_expert_variant="expert",
        dtype="float32",
        action_dim=4,
        action_horizon=3,
        pytorch_compile_mode=None,
        velocity_mode="potential_dot",
        freeze_vlm=True,
    )
    model = PI0Pytorch(config)
    model.configure_expert_only(vlm_dtype=torch.float32)
    images = [torch.randn(2, 3, 14, 14)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    masks = torch.ones_like(tokens, dtype=torch.bool)
    observation = SimpleNamespace(state=torch.zeros(2, 4))
    monkeypatch.setattr(
        model,
        "_preprocess_observation",
        lambda obs, *, train: (images, image_masks, tokens, masks, obs.state),
    )
    return model, observation


def test_actual_expert_double_backward_and_frozen_vlm(tiny_model):
    model, observation = tiny_model
    model.train()
    actions = torch.randn(2, 3, 4)
    loss = model(observation, actions, noise=torch.randn_like(actions), time=torch.tensor([0.3, 0.7])).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.expert_modules().named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())
    assert not model.paligemma_with_expert.paligemma.training


def test_bfloat16_frozen_vlm_and_float32_expert_support_double_backward(tiny_model):
    model, observation = tiny_model
    model.configure_expert_only(vlm_dtype=torch.bfloat16)
    model.train()
    actions = torch.randn(2, 3, 4)
    loss = model(observation, actions, noise=torch.randn_like(actions), time=torch.tensor([0.3, 0.7])).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.expert_modules().named_parameters():
        assert parameter.dtype == torch.float32, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_native_cache_is_immutable_and_conditioning_is_used(tiny_model):
    model, observation = tiny_model
    model.eval()
    context = model.encode_context(observation)
    cache_before = [(key.clone(), value.clone()) for key, value in context.past_key_values]
    x = torch.randn(2, 3, 4, requires_grad=True)
    time = torch.tensor([0.4, 0.7])
    first = model.predict_velocity(x, time, context, create_graph=True)
    second = model.predict_velocity(x, time, context, create_graph=True)
    torch.testing.assert_close(first, second)
    for (key, value), (old_key, old_value) in zip(context.past_key_values, cache_before, strict=True):
        torch.testing.assert_close(key, old_key, rtol=0, atol=0)
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)
    # Permute only the observation cache, leaving actions/time unchanged.
    from transformers.cache_utils import DynamicCache

    from openpi.models_pytorch.pi0_pytorch import ActionContext

    swapped = DynamicCache.from_legacy_cache(tuple((k.flip(0), v.flip(0)) for k, v in cache_before))
    other = ActionContext(context.state, context.prefix_pad_masks, swapped)
    changed = model.predict_velocity(x, time, other, create_graph=True)
    assert not torch.allclose(first, changed, atol=1e-7, rtol=1e-6)


def test_checkpointed_suffix_preserves_output_and_parameter_gradients(tiny_model):
    model, observation = tiny_model
    model.train()
    actions = torch.randn(2, 3, 4)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.25, 0.75])
    reference = model(observation, actions, noise=noise, time=time).mean()
    reference.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable()
    checked = model(observation, actions, noise=noise, time=time).mean()
    checked.backward()
    torch.testing.assert_close(reference, checked)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name])


def test_sampling_restores_local_gradients_without_parameter_gradients(tiny_model):
    model, observation = tiny_model
    model.eval()
    noise = torch.randn(2, 3, 4)
    with torch.no_grad():
        result = model.sample_actions("cpu", observation, noise=noise, num_steps=10)
        again = model.sample_actions("cpu", observation, noise=noise, num_steps=10)
    assert not result.requires_grad
    torch.testing.assert_close(result, again, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    model.requires_grad_(requires_grad=False)  # Policy serving freezes parameters, not input derivatives.
    frozen = model.sample_actions("cpu", observation, noise=noise, num_steps=10)
    torch.testing.assert_close(result, frozen, rtol=0, atol=0)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="input gradients"):
        model.sample_actions("cpu", observation, noise=noise)


def test_joint_vector_path_matches_cached_vector_path(tiny_model):
    model, observation = tiny_model
    model.eval()
    model.velocity_mode = "direct"
    actions = torch.randn(2, 3, 4)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.8])
    cached = model(observation, actions, noise=noise, time=time)
    model.freeze_vlm = False  # exercise upstream joint computation, unchanged weights
    joint = model(observation, actions, noise=noise, time=time)
    torch.testing.assert_close(cached, joint, atol=2e-5, rtol=2e-4)


def test_zero_head_controls_match_and_begin_learning(tiny_model):
    model, observation = tiny_model
    model.zero_action_output()
    context = model.encode_context(observation)
    x = torch.randn(2, 3, 4, requires_grad=True)
    time = torch.tensor([0.2, 0.8])
    potential = model.predict_velocity(x, time, context, create_graph=True)
    model.velocity_mode = "direct"
    direct = model.predict_velocity(x, time, context)
    torch.testing.assert_close(potential, direct, rtol=0, atol=0)
    assert torch.count_nonzero(potential) == 0
    (potential - 1).square().mean().backward()
    assert torch.count_nonzero(model.action_out_proj.weight.grad) > 0
