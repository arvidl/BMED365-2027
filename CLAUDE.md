# BMED365-2027 – rules for AI coding agents

Course material (Jupyter notebooks, slides, READMEs) for BMED365 at the University of Bergen, Spring 2027. This is not a Python package.

The repo is developed on two machines (a MacBook Pro M4 Max and a Dell Precision with an NVIDIA RTX A5000 running Ubuntu) and by more than one agent (Claude Code, and Cursor with other models). Git is the only thing that keeps them in sync, so the rules below matter.

## The repo is public

- https://github.com/arvidl/BMED365-2027 is public. Everything pushed is visible to students immediately and should be treated as permanently public.
- Never commit secrets, API keys, exam questions, unreleased solutions, or data that cannot be shared (e.g. patient images).
- API keys live in `.env` (git-ignored), never in notebooks or code.

## Branches and syncing

- Work on `dev`. `main` is what students and the "Open in Colab" badges see.
- Merge `dev` into `main` only when the user asks, e.g. to test in Colab or release material.
- Run `git pull` before starting work. The other machine or another agent may have pushed.
- Commit and push only when the user asks. Show the diff summary before pushing.
- Never force-push or rewrite history unless the user explicitly asks.
- Assume another agent may be editing this working tree. If files change unexpectedly, stop and tell the user rather than overwriting.

## Notebooks

- Running a notebook rewrites its outputs, execution counts and metadata. Do not commit a notebook that was only run as a test; discard the changes with `git restore <notebook>`.
- To test a notebook without touching it, execute it into a directory outside the repo:
  `uv run jupyter nbconvert --to notebook --execute <nb>.ipynb --ExecutePreprocessor.kernel_name=python3 --output-dir <scratch dir>`
- Commit changed outputs only when updating them is the purpose of the change.
- Do not commit changes to a notebook's `kernelspec` metadata. Editors such as Cursor rewrite it when a kernel is selected (e.g. `"name": "python3"` becomes `"name": "bmed365-2027"` or `.venv`), and Colab only knows `python3`.
- Many notebooks contain large embedded images. Do not print or read whole notebooks into context; search for the specific cell.
- When editing notebook text, change only the lines involved. Do not reformat or re-serialize the JSON.

## Large files

- GitHub rejects files over 100 MB. `Lab4-Comp-Imag/01-imaging-intro.ipynb` is at 99 MB; check its size before committing any change to it.
- Do not add new files over 50 MB. Datasets and model weights should be downloaded at runtime and listed in `.gitignore`.
- The repo is already over 1 GB. Do not add bulk data.

## Python environment (uv)

- The environment is defined by `pyproject.toml` and `uv.lock`, with Python 3.12. Set it up with `uv sync`.
- Run things with `uv run ...` (e.g. `uv run jupyter lab`, `uv run python script.py`).
- Never `pip install` or `conda install` into `.venv`. Add packages with `uv add <package>` and commit `pyproject.toml` and `uv.lock` together.
- The lockfile must keep resolving for macOS Apple Silicon, Linux (CUDA) and Intel Macs. Intel Macs are pinned to older torch/TensorFlow through platform markers in `pyproject.toml`; do not remove those markers.
- `.venv/` and the Jupyter kernel are per machine and not in git. Register the kernel once per machine:
  `uv run python -m ipykernel install --user --name bmed365-2027`
- Three sub-projects keep their own separate conda environments and are not part of the uv environment: PyCaret (`Lab0-ML/pycaret-environment.yml`), IMC (`Lab4-Comp-Imag/IMC-example/environment-imc.yml`) and MLX (`Lab5-Comp-Mod/MLX-Bio-Qwen`).

## Platforms

- Students run the notebooks either locally (macOS Apple Silicon or Linux, see `README.md`) or in Google Colab. Code must work in both.
- Pick the device at runtime (CUDA, then MPS, then CPU). Do not hard-code one.
- Do not hard-code absolute or machine-specific paths.
- The development machines have 128 GB RAM. A free Colab session has about 12 GB RAM and a 16 GB T4 GPU, so a notebook that runs locally can still fail in Colab. Keep memory use modest and say when something has not been tested in Colab.
- Links and Colab badges use `https://github.com/arvidl/BMED365-2027/blob/main/...`.

## Writing

- Course material is in English and written for biomedicine students, many with little programming experience. Explain steps plainly.
- Update the relevant lab `README.md` when notebooks are added, renamed or removed.
