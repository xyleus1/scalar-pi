"""Shared fixtures; heavy model dependencies load only for model tests."""

import pytest


@pytest.fixture
def tiny_model(monkeypatch):
    from types import SimpleNamespace

    import torch
    from transformers import GemmaForCausalLM
    from transformers import PaliGemmaForConditionalGeneration
    from transformers import SiglipVisionConfig

    from openpi.models import gemma
    from openpi.models_pytorch import gemma_pytorch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

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
