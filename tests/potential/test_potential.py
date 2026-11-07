"""CPU mathematical checks; these do not establish full-model CUDA readiness."""

import json
import math
from pathlib import Path
import runpy

import pytest
import torch

from openpi.models_pytorch.potential import dot_potential
from openpi.models_pytorch.potential import potential_velocity


def test_linear_field_has_full_symmetric_jacobian_term():
    generator = torch.Generator().manual_seed(7)
    actions = torch.randn(2, 2, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    matrix = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    bias = torch.randn(6, generator=generator, dtype=torch.float64)

    def field(x):
        return (x.flatten(1) @ matrix.T + bias).reshape_as(x)

    velocity = potential_velocity(field, actions, create_graph=True)
    expected = (actions.flatten(1) @ (matrix + matrix.T).T + bias).reshape_as(actions)
    torch.testing.assert_close(velocity, expected)


def test_double_precision_gradient_and_double_gradient():
    generator = torch.Generator().manual_seed(11)
    actions = torch.randn(1, 2, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 2, generator=generator, dtype=torch.float64, requires_grad=True)

    def velocity(x, w):
        return potential_velocity(lambda value: torch.tanh(value @ w.T), x, create_graph=True)

    assert torch.autograd.gradcheck(velocity, (actions, weight))
    assert torch.autograd.gradgradcheck(velocity, (actions, weight))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_contraction_precision_and_sum_not_mean(dtype):
    actions = torch.ones(2, 3, 4, dtype=dtype)
    field = torch.full_like(actions, 2.0)
    result = dot_potential(actions, field)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert result.dtype == expected_dtype
    torch.testing.assert_close(result, torch.full((2,), 24.0, dtype=expected_dtype))


def test_zero_output_head_can_learn_then_updates_trunk():
    torch.manual_seed(17)
    trunk = torch.nn.Linear(2, 4, dtype=torch.float64)
    head = torch.nn.Linear(4, 2, dtype=torch.float64)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.SGD([*trunk.parameters(), *head.parameters()], lr=0.1)
    actions = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)

    def field(x):
        return head(torch.tanh(trunk(x)))

    initial = potential_velocity(field, actions, create_graph=True)
    torch.testing.assert_close(initial, torch.zeros_like(initial))
    (initial - 1).square().mean().backward()
    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum() > 0
    assert trunk.weight.grad is not None
    assert torch.count_nonzero(trunk.weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated = potential_velocity(field, actions, create_graph=True)
    (updated - 1).square().mean().backward()
    assert trunk.weight.grad is not None
    assert trunk.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("steps", [1, 10])
def test_reverse_time_euler_sign_for_deterministic_action(steps):
    target = torch.tensor([[[0.25, -0.75]]], dtype=torch.float64)
    state = torch.tensor([[[1.5, 0.5]]], dtype=torch.float64)
    with torch.no_grad():
        for index in range(steps):
            time = 1 - index / steps
            with torch.enable_grad():
                leaf = state.detach().requires_grad_(requires_grad=True)
                velocity = potential_velocity(
                    lambda x, current_time=time: (x - 2 * target) / (2 * current_time), leaf, create_graph=False
                )
            state = (state - velocity / steps).detach()
    torch.testing.assert_close(state, target)


def test_gaussian_mixture_velocity_and_energy_identity():
    """Use an analytic scalar to check signs on a genuinely multimodal path.

    The radial field below is a test-only realization of that known scalar on
    nonzero inputs. Production uses the complete pretrained action expert.
    """
    time = 0.37
    data_variance = 0.16
    means = torch.tensor([[-1.5, 0.5], [1.5, -0.5]], dtype=torch.float64)
    weights = torch.tensor([0.3, 0.7], dtype=torch.float64)
    actions = torch.tensor([[[0.8, 0.4]], [[-0.7, 0.2]]], dtype=torch.float64, requires_grad=True)
    variance = (1 - time) ** 2 * data_variance + time**2

    def log_density(x):
        offsets = x[:, 0, None, :] - (1 - time) * means
        component_logs = weights.log() - offsets.square().sum(-1) / (2 * variance)
        component_logs = component_logs - math.log(2 * math.pi * variance)
        return torch.logsumexp(component_logs, dim=-1)

    def scalar(x):
        return (-time * log_density(x) - 0.5 * x.square().sum((-2, -1))) / (1 - time)

    zero_scalar = scalar(torch.zeros_like(actions)).detach()

    def field(x):
        coefficient = (scalar(x) - zero_scalar) / x.square().sum((-2, -1))
        return x * coefficient[:, None, None]

    velocity = potential_velocity(field, actions, create_graph=True)
    offsets = actions[:, 0, None, :] - (1 - time) * means
    responsibilities = torch.softmax(weights.log() - offsets.square().sum(-1) / (2 * variance), dim=-1)
    posterior_means = means + data_variance * (1 - time) / variance * offsets
    posterior_mean = (responsibilities[..., None] * posterior_means).sum(1)[:, None, :]
    torch.testing.assert_close(velocity, (actions - posterior_mean) / time)
    score = torch.autograd.grad(log_density(actions).sum(), actions, retain_graph=True)[0]
    torch.testing.assert_close(score, -(actions + (1 - time) * velocity) / time)

    potential = dot_potential(actions, field(actions))
    energy = (actions.square().sum((-2, -1)) + 2 * (1 - time) * potential) / (2 * time)
    negative_score = torch.autograd.grad(energy.sum(), actions)[0]
    torch.testing.assert_close(negative_score, -score)


def test_one_step_exact_mixture_field_returns_conditional_mean():
    mean_action = torch.tensor([[[0.4, -0.3]]], dtype=torch.float64)
    noise = torch.tensor([[[2.0, 1.2]], [[-0.8, 0.7]]], dtype=torch.float64, requires_grad=True)
    # At tau=1 the entire action distribution enters only through its mean.
    velocity = potential_velocity(lambda x: 0.5 * x - mean_action, noise, create_graph=False)
    torch.testing.assert_close(noise - velocity, mean_action.expand_as(noise))


def test_batch_independence_and_hessian_symmetry():
    generator = torch.Generator().manual_seed(23)
    actions = torch.randn(2, 2, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4, 4, generator=generator, dtype=torch.float64)

    def field(x):
        return torch.tanh(x.flatten(1) @ weight.T).reshape_as(x)

    velocity = potential_velocity(field, actions, create_graph=True)
    for index in range(actions.shape[0]):
        alone = actions[index : index + 1].detach().requires_grad_(requires_grad=True)
        torch.testing.assert_close(velocity[index : index + 1], potential_velocity(field, alone, create_graph=False))

    first = torch.randn(actions.shape, generator=generator, dtype=torch.float64)
    second = torch.randn(actions.shape, generator=generator, dtype=torch.float64)
    jacobian_first = torch.autograd.grad((velocity * first).sum(), actions, retain_graph=True)[0]
    jacobian_second = torch.autograd.grad((velocity * second).sum(), actions)[0]
    torch.testing.assert_close((jacobian_first * second).sum(), (jacobian_second * first).sum())


def test_inference_wrapper_does_not_retain_graph_or_parameter_gradients():
    head = torch.nn.Linear(2, 2, dtype=torch.float64)
    state = torch.ones(1, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(100):
            with torch.enable_grad():
                leaf = state.detach().requires_grad_(requires_grad=True)
                velocity = potential_velocity(head, leaf, create_graph=False)
            state = (state - 0.01 * velocity).detach()
            assert velocity.grad_fn is None
            assert not state.requires_grad
            assert state.grad_fn is None
    assert all(parameter.grad is None for parameter in head.parameters())


def test_invalid_shape_and_dtype():
    with pytest.raises(ValueError, match="shape"):
        dot_potential(torch.ones(2, 3), torch.ones(2, 3))
    with pytest.raises(ValueError, match="shape"):
        dot_potential(torch.ones(1, 2, 3), torch.ones(1, 2, 4))
    with pytest.raises(TypeError, match="floating-point"):
        dot_potential(torch.ones(1, 2, 3, dtype=torch.int64), torch.ones(1, 2, 3))


def test_caller_must_manage_gradient_context():
    actions = torch.ones(1, 2, 2)
    with pytest.raises(ValueError, match="action leaf"):
        potential_velocity(torch.tanh, actions, create_graph=False)
    actions.requires_grad_(requires_grad=True)
    with torch.no_grad(), pytest.raises(RuntimeError, match="enable_grad"):
        potential_velocity(torch.tanh, actions, create_graph=False)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        potential_velocity(torch.tanh, actions, create_graph=False)


@pytest.fixture
def preflight_module():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/potential_preflight.py"))


@pytest.fixture
def measured_profile():
    # Deliberately tiny test numbers: this fixture is not a measured run report.
    return {
        "measured_gpu": True,
        "model_kind": "pi05_full",
        "hardware": "unit-test fixture",
        "measurement_source": "unit-test fixture, not a real profiler log",
        "warmup_optimizer_updates": 10,
        "timed_optimizer_updates": 100,
        "effective_batch_size": 256,
        "optimizer_updates": 30_000,
        "training_seeds": 3,
        "seconds_per_optimizer_update": {"b1": 1.0, "potential": 2.0},
        "evaluation_episodes": 2000,
        "episodes_per_gpu_hour": 100.0,
        "fixed_cost_usd": 100.0,
    }


def test_budget_estimate_counts_accumulated_updates_and_evaluation(preflight_module, measured_profile):
    estimate = preflight_module["estimate_budget"](measured_profile, 2.0, 2000.0)
    assert estimate["training_gpu_hours"] == 75.0
    assert estimate["evaluation_gpu_hours"] == 20.0
    assert estimate["projected_total_usd"] == 328.0
    assert estimate["within_budget"]
    assert not preflight_module["estimate_budget"](measured_profile, 2.0, 300.0)["within_budget"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measured_gpu", False, "measured_gpu"),
        ("model_kind", "synthetic", "pi05_full"),
        ("timed_optimizer_updates", 5, "100 warmed"),
        ("effective_batch_size", 1, "effective_batch_size=256"),
        ("training_seeds", 1.5, "integer"),
        ("synthetic_input", True, "Synthetic"),
        ("valid_training_cost_profile", False, "Synthetic"),
    ],
)
def test_budget_rejects_unqualified_measurements(preflight_module, measured_profile, field, value, message):
    measured_profile[field] = value
    with pytest.raises(ValueError, match=message):
        preflight_module["estimate_budget"](measured_profile, 2.0, 2000.0)


def test_research_asset_check_does_not_accept_shape_only_checkpoint(preflight_module, tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"placeholder, not actual weights")
    result = preflight_module["inspect_assets"](tmp_path, None, None, "research")
    assert result["checkpoint"]["weights_present"]
    assert not result["checkpoint"]["metadata_valid"]
    assert not result["checkpoint"]["ready"]


def test_qualification_help_is_available_without_loading_a_model(capsys):
    qualifier = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/qualify_potential.py"))
    with pytest.raises(SystemExit) as stopped:
        qualifier["parse_args"](["--help"])
    assert stopped.value.code == 0
    assert "--checkpoint-kind" in capsys.readouterr().out


def test_qualification_rejects_cpu_and_never_claims_a_profile(tmp_path, monkeypatch):
    qualifier = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/qualify_potential.py"))
    (tmp_path / "model.safetensors").write_bytes(b"never loaded on a CPU-only diagnostic")
    report_path = tmp_path / "qualification.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    status = qualifier["main"](["--checkpoint", str(tmp_path), "--report", str(report_path)])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert status == 1
    assert report["synthetic_input"]
    assert not report["correctness_checks_passed"]
    assert not report["measured_gpu"]
    assert not report["valid_training_cost_profile"]
    assert not report["valid_benchmark_evaluation"]
    assert "CUDA" in report["error"]["message"]
    with pytest.raises(SystemExit) as stopped:
        qualifier["parse_args"](["--checkpoint", str(tmp_path), "--report", str(report_path)])
    assert stopped.value.code == 2
