"""Build the paper with portable Python and TeX executables."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "build/scalar-pi-study.pdf")
    args = parser.parse_args()
    output = args.output.resolve()
    build = HERE / "build/tex"
    build.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "build_assets.py"], cwd=HERE, check=True)
    tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build/tex", "manuscript.tex"]
    for command in (tex, ["bibtex", "build/tex/manuscript"], tex, tex):
        subprocess.run(command, cwd=HERE, check=True)
    shutil.copyfile(build / "manuscript.pdf", output)
    print(output)


if __name__ == "__main__":
    main()
