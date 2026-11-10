"""Exercise the actual Policy wrapper with a tiny CPU model, without checkpoints."""

import types

import numpy as np
import pytest
import torch

from openpi.policies.policy import Policy


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(action_horizon=10, action_dim=32)
        self.last_noise = None
        self.last_num_steps = None

    def sample_actions(self, device, observation, *, noise=None, num_steps=10):
        assert observation.state.shape == (1, 7)
        if noise is None:
            noise = torch.randn(1, self.config.action_horizon, self.config.action_dim, device=device)
        self.last_noise = noise.clone()
        self.last_num_steps = num_steps
        return noise + num_steps


def _observation(seed=123):
    return {"image": {}, "image_mask": {}, "state": np.zeros(7, dtype=np.float32), "noise_seed": seed}


def _policy(num_steps=10):
    model = RecordingModel()

    def validate_request_cleanup(inputs):
        assert "noise_seed" not in inputs
        return inputs

    policy = Policy(
        model,
        is_pytorch=True,
        pytorch_device="cpu",
        transforms=[validate_request_cleanup],
        sample_kwargs={"num_steps": num_steps},
    )
    return policy, model


def test_seeded_policy_is_repeatable_without_mutating_global_rng_or_input():
    policy, _ = _policy()
    request = _observation()
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    first = policy.infer(request)
    policy.infer(_observation(456))
    repeated = policy.infer(request)
    np.testing.assert_array_equal(first["actions"], repeated["actions"])
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    new_numpy_state = np.random.get_state()
    assert numpy_state[0] == new_numpy_state[0]
    np.testing.assert_array_equal(numpy_state[1], new_numpy_state[1])
    assert numpy_state[2:] == new_numpy_state[2:]
    assert request["noise_seed"] == 123
    np.testing.assert_array_equal(request["state"], np.zeros(7))


def test_k_endpoints_receive_identical_explicit_noise_and_honor_solver_kwargs():
    one, model_one = _policy(num_steps=1)
    ten, model_ten = _policy(num_steps=10)
    output_one = one.infer(_observation(2**32 - 1))
    output_ten = ten.infer(_observation(2**32 - 1))
    torch.testing.assert_close(model_one.last_noise, model_ten.last_noise)
    assert model_one.last_noise.shape == (1, 10, 32)
    assert model_one.last_num_steps == 1
    assert model_ten.last_num_steps == 10
    np.testing.assert_allclose(output_ten["actions"] - output_one["actions"], 9.0, atol=1e-6)
    assert output_one["policy_timing"]["total_ms"] >= output_one["policy_timing"]["infer_ms"] >= 0
    assert "cuda_ms" not in output_one["policy_timing"]


def test_request_seed_matches_explicit_noise_and_distinct_seeds_change_noise():
    policy, model = _policy()
    seeded = policy.infer(_observation(99))
    expected_noise = torch.randn(10, 32, generator=torch.Generator().manual_seed(99)).numpy()
    request = _observation()
    request.pop("noise_seed")
    explicit = policy.infer(request, noise=expected_noise)
    np.testing.assert_array_equal(seeded["actions"], explicit["actions"])
    first_noise = model.last_noise.clone()
    policy.infer(_observation(100))
    assert not torch.equal(first_noise, model.last_noise)


@pytest.mark.parametrize("seed", [-1, 2**32, "7", 7.0, True])
def test_policy_rejects_invalid_request_seeds(seed):
    policy, _ = _policy()
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        policy.infer(_observation(seed))


def test_policy_rejects_ambiguous_explicit_noise_and_seed():
    policy, _ = _policy()
    with pytest.raises(ValueError, match="either noise or noise_seed"):
        policy.infer(_observation(), noise=np.zeros((10, 32), dtype=np.float32))


@pytest.mark.parametrize("bad_shape", [False, True])
def test_known_invalid_actions_are_policy_results_not_transport_exceptions(bad_shape):
    class InvalidModel(RecordingModel):
        def sample_actions(self, device, observation, **kwargs):
            return torch.zeros(1, 2, 32) if bad_shape else torch.full((1, 10, 32), float("nan"))

    policy = Policy(InvalidModel(), is_pytorch=True, metadata={"research": {"schema_version": 1}})
    result = policy.infer(_observation())
    assert result["policy_error"]["type"] == "invalid_actions"
    assert "actions" not in result
    assert result["policy_timing"]["total_ms"] >= result["policy_timing"]["infer_ms"]


def test_unexpected_model_exception_is_not_reclassified_as_a_policy_failure():
    class BrokenModel(RecordingModel):
        def sample_actions(self, device, observation, **kwargs):
            raise RuntimeError("unclassified backend failure")

    policy = Policy(BrokenModel(), is_pytorch=True, metadata={"research": {"schema_version": 1}})
    with pytest.raises(RuntimeError, match="unclassified"):
        policy.infer(_observation())
