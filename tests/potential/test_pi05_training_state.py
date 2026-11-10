"""Test accumulation and exact restart on the real, reduced-size π0.5 graph."""

import copy

import pytest
from test_pi05_integration import tiny_model  # noqa: F401 -- shared production-model fixture.
import torch
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.pi0_pytorch import ActionContext
from openpi.training.potential import ResumableBatchSampler
from openpi.training.potential import load_training_checkpoint
from openpi.training.potential import make_ema
from openpi.training.potential import save_checkpoint
from openpi.training.potential import stable_seed
from openpi.training.potential import torch_rng_scope


@pytest.mark.parametrize("mode", ["direct", "potential_dot"])
def test_actual_pi05_accumulation_matches_full_preprocessed_batch(tiny_model, mode):  # noqa: F811
    model, observation = tiny_model
    model.velocity_mode = mode
    model.train()
    context = model.encode_context(observation)
    x = torch.randn(2, 3, 4)
    target = torch.randn_like(x)
    time = torch.tensor([0.3, 0.7])
    full = model.predict_velocity(x.clone().requires_grad_(), time, context, create_graph=True)
    (full - target).square().mean().backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    for index in range(2):
        cache = DynamicCache.from_legacy_cache(
            tuple((k[index : index + 1], v[index : index + 1]) for k, v in context.past_key_values)
        )
        single = ActionContext(context.state[index : index + 1], context.prefix_pad_masks[index : index + 1], cache)
        value = model.predict_velocity(
            x[index : index + 1].clone().requires_grad_(), time[index : index + 1], single, create_graph=True
        )
        ((value - target[index : index + 1]).square().mean() / 2).backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, expected[name], atol=1e-4, rtol=1e-4)


def test_actual_pi05_restart_preserves_samples_corruption_optimizer_and_ema(tiny_model, tmp_path):  # noqa: F811
    model, observation = tiny_model
    model.train()
    initial = copy.deepcopy(model)
    actions = torch.linspace(-1, 1, 8 * 3 * 4).reshape(8, 3, 4)
    optimizer = torch.optim.AdamW(model.expert_modules().parameters(), lr=1e-3)
    ema = make_ema(model.expert_modules())
    signature = {"seed": 42, "optimizer": {"lr": 1e-3}, "source_snapshot_sha256": "a" * 64}

    def step(network, opt, averaged, update, cursor):
        sampler = iter(ResumableBatchSampler(8, 2, 4, 42, start_microbatch=cursor))
        loader = iter(torch.utils.data.DataLoader(actions, batch_sampler=sampler, num_workers=0))
        identity = iter(ResumableBatchSampler(8, 2, 4, 42, start_microbatch=cursor))
        trace = []
        opt.zero_grad(set_to_none=True)
        for micro in range(2):
            batch = next(loader)
            indices = next(identity)
            with torch_rng_scope(stable_seed(42, "corruption", update, micro), torch.device("cpu")):
                noise = network.sample_noise(batch.shape, "cpu")
                time = network.sample_time(len(batch), "cpu")
            with torch_rng_scope(stable_seed(42, "augmentation", update, micro), torch.device("cpu")):
                (network(observation, batch, noise=noise, time=time).mean() / 2).backward()
            trace.append((indices, batch.clone(), noise.clone(), time.clone()))
        torch.nn.utils.clip_grad_norm_(network.expert_modules().parameters(), 1.0)
        opt.step()
        averaged.update_parameters(network.expert_modules())
        return trace

    step(model, optimizer, ema, 0, 0)
    metadata = {"velocity_mode": "potential_dot", "action_horizon": 3, "action_dim": 4, "resume_signature": signature}
    checkpoint = save_checkpoint(
        tmp_path / "1",
        model=model,
        expert=model.expert_modules(),
        optimizer=optimizer,
        ema=ema,
        state={"optimizer_updates": 1, "consumed_microbatches": 2, "resume_signature": signature},
        metadata=metadata,
    )
    expected_trace = step(model, optimizer, ema, 1, 2)
    resumed = initial
    resumed_opt = torch.optim.AdamW(resumed.expert_modules().parameters(), lr=1e-3)
    resumed_ema = make_ema(resumed.expert_modules())
    state = load_training_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_opt,
        ema=resumed_ema,
        velocity_mode="potential_dot",
        action_horizon=3,
        action_dim=4,
        expected_resume_signature=signature,
    )
    actual_trace = step(resumed, resumed_opt, resumed_ema, state["optimizer_updates"], state["consumed_microbatches"])
    for expected, actual in zip(expected_trace, actual_trace, strict=True):
        assert expected[0] == actual[0]
        for a, b in zip(expected[1:], actual[1:], strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
    for key, value in ema.state_dict().items():
        torch.testing.assert_close(value, resumed_ema.state_dict()[key], atol=0, rtol=0)
    for key, expected in optimizer.state_dict()["state"].items():
        for name, value in expected.items():
            torch.testing.assert_close(value, resumed_opt.state_dict()["state"][key][name], atol=0, rtol=0)
