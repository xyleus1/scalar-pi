"""Install the exact OpenPI Transformers replacement into the active environment."""

from pathlib import Path
import shutil

import transformers


def main():
    if transformers.__version__ != "4.53.2":
        raise RuntimeError("OpenPI replacements require Transformers 4.53.2")
    source = Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    target = Path(transformers.__file__).parent
    for item in source.rglob("*.py"):
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
    print(f"Installed the pinned OpenPI replacement into {target}")


if __name__ == "__main__":
    main()
