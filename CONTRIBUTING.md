# Contributing to Scalar-Pi

For bugs and proposed changes, use the [Scalar-Pi issue tracker](https://github.com/xyleus1/scalar-pi/issues). Include a small reproduction, the source revision, and relevant dependency versions. Remove credentials and private paths from logs before sharing them.

Keep changes focused. Model changes should state which experiment they implement; infrastructure or organization changes should preserve the existing model and checkpoint behavior.

## Development checks

Install the development environment described in the [README](README.md). Before submitting a change:

```bash
uv run --no-sync pre-commit run --all-files
uv run --no-sync python -m pytest -q tests/potential
```

The CPU CI environment is separate from the canonical GPU environment. See the research workflow for its pinned installation commands. CPU tests cover gradients, small production-model fixtures, checkpoint recovery, data identities, and analysis; they do not establish full-model hardware compatibility.

Keep existing scientific contracts intact unless the change explicitly introduces a new experiment: checkpoint keys and metadata, dtype policies, input gradients, RNG restoration, optimizer/EMA order, pairing, and statistical estimators. Add a regression test when fixing a behavioral bug. Documentation-only changes do not need new tests.

Use short descriptions that explain the behavior changed and how it was checked. Preserve upstream attribution and license notices. The project maintainer is Nima Kamali ([xyleus1](https://github.com/xyleus1)).
