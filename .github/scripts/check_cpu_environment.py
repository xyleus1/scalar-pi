"""Fail CI if its isolated test environment drifts into a GPU or RLDS stack."""

import importlib.metadata
from pathlib import Path
import platform
import sys

import jax
import torch
import transformers


def main():
    assert sys.version_info[:3] == (3, 11, 9), sys.version
    assert platform.system() == "Linux", platform.platform()
    expected = {
        "torch": "2.7.1+cpu",
        "torchvision": "0.22.1+cpu",
        "jax": "0.5.3",
        "jaxlib": "0.5.3",
        "transformers": "4.53.2",
        "numpy": "1.26.4",
    }
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        assert actual == version, (name, actual, version)
    installed = {dist.metadata["Name"].lower().replace("_", "-") for dist in importlib.metadata.distributions()}
    forbidden = sorted(name for name in installed if name.startswith(("nvidia-", "jax-cuda", "triton", "tensorflow")))
    assert not forbidden, forbidden
    assert torch.version.cuda is None, torch.version.cuda
    assert jax.default_backend() == "cpu", jax.devices()
    source = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/transformers_replace"
    target = Path(transformers.__file__).parent
    for item in source.rglob("*.py"):
        assert item.read_bytes() == (target / item.relative_to(source)).read_bytes(), item
    print("Pinned CPU versions and the exact Transformers replacement verified.")


if __name__ == "__main__":
    main()
