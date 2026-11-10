"""Exercise audit failures without downloading robot data or tokenizers."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.research.data import audit_libero_dataset


@pytest.fixture
def audit_fixture():
    counts = {"total_frames": 6, "total_episodes": 2, "total_tasks": 2}
    actions = np.arange(42, dtype=np.float32).reshape(6, 7)
    samples = []
    for i in range(6):
        episode, frame = divmod(i, 3)
        indices = np.minimum(np.arange(i, i + 10), (episode + 1) * 3 - 1)
        samples.append(
            {
                "index": i,
                "episode_index": episode,
                "frame_index": frame,
                "timestamp": frame / 10,
                "task_index": episode,
                "prompt": f"task {episode}",
                "state": np.zeros(8, dtype=np.float32),
                "image": np.zeros((3, 256, 256), dtype=np.float32),
                "wrist_image": np.ones((3, 256, 256), dtype=np.float32),
                "actions": actions[indices],
                "actions_is_pad": np.arange(i, i + 10) >= (episode + 1) * 3,
            }
        )

    class Dataset:
        def __init__(self):
            self.meta = SimpleNamespace(info={**counts, "fps": 10}, tasks={0: "task 0", 1: "task 1"})
            self.episode_data_index = {"from": np.array([0, 3]), "to": np.array([3, 6])}
            self.hf_dataset = [{"actions": action} for action in actions]

        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            return samples[i]

    views = {}
    for name in ("pi05_libero", "pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential"):
        views[name] = [
            {
                "actions": np.pad(sample["actions"], ((0, 0), (0, 25))),
                "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False},
                "tokenized_prompt": np.array([1, 2]),
            }
            for sample in samples
        ]
    return Dataset(), views, samples, counts


def test_audit_decodes_all_windows_checks_boundaries_and_exports_safe_corpus(audit_fixture, tmp_path):
    dataset, variants, _, counts = audit_fixture
    output = tmp_path / "corpus.npz"
    result = audit_libero_dataset(
        dataset,
        variants,
        expected_counts=counts,
        corpus_output=output,
        corpus_metadata={"revision": "a" * 40},
    )
    assert result["decoded_windows"] == 6
    assert result["task_window_counts"] == {0: 3, 1: 3}
    assert len(result["boundary_fixtures"]) == 6
    with np.load(output, allow_pickle=False) as corpus:
        assert corpus["images"].shape == (2, 256, 256, 3)
        assert corpus["images"].dtype == np.uint8
        assert corpus["wrist_images"].max() == 255
        assert corpus["noise"].shape == (2, 10, 32)
        assert corpus["prompts"].tolist() == ["task 0", "task 1"]
        assert corpus["dataset_indices"].tolist() == [1, 4]
        assert json.loads(str(corpus["metadata_json"])) == {"revision": "a" * 40}


@pytest.mark.parametrize("failure", ["cross_episode", "prompt", "nonfinite", "image", "transform", "padding"])
def test_audit_rejects_semantically_wrong_data(audit_fixture, failure):
    dataset, variants, samples, counts = audit_fixture
    if failure == "cross_episode":
        samples[2]["actions"][1] = 1000
    elif failure == "prompt":
        samples[1]["prompt"] = "wrong"
    elif failure == "nonfinite":
        samples[1]["state"][0] = np.nan
    elif failure == "image":
        samples[1]["image"] = np.zeros((3, 224, 224))
    elif failure == "transform":
        variants["pi05_libero_b1"][1]["tokenized_prompt"][0] = 42
    else:
        samples[2]["actions_is_pad"][1] = False
    with pytest.raises((ValueError, AssertionError)):
        audit_libero_dataset(dataset, variants, expected_counts=counts)


def test_official_libero_input_and_normalization_transforms_match_research_configs(monkeypatch):
    from openpi.models import tokenizer
    from openpi.shared.normalize import NormStats
    from openpi.training import config
    from openpi.training import data_loader

    class Tokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def tokenize(self, prompt, state=None):
            return np.array([1, len(prompt), 2]), np.ones(3, dtype=bool)

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", Tokenizer)
    raw = {
        "image": np.full((3, 256, 256), 0.5, dtype=np.float32),
        "wrist_image": np.ones((3, 256, 256), dtype=np.float32),
        "state": np.arange(8, dtype=np.float32),
        "actions": np.arange(70, dtype=np.float32).reshape(10, 7) / 70,
        "prompt": "pick up the cup",
    }
    norm = {
        key: NormStats(mean=np.zeros(size), std=np.ones(size), q01=-np.ones(size), q99=np.ones(size))
        for key, size in (("state", 8), ("actions", 7))
    }
    reference = None
    for name in ("pi05_libero", "pi05_libero_b0", "pi05_libero_b1", "pi05_libero_potential"):
        cfg = config.get_config(name)
        import dataclasses

        data = dataclasses.replace(cfg.data.create(cfg.assets_dirs, cfg.model), norm_stats=norm)
        value = data_loader.transform_dataset([copy.deepcopy(raw)], data)[0]
        if reference is None:
            reference = value
        else:
            import jax

            jax.tree.map(np.testing.assert_array_equal, reference, value)
    assert reference["actions"].shape == (10, 32)
    assert np.count_nonzero(reference["actions"][..., 7:]) == 0
    assert not reference["image_mask"]["right_wrist_0_rgb"]
