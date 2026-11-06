## Project

Graduate class project (CIS 6930 Generative AI, USF). Primary paper: MaPLe: Multi-modal Prompt
Learning (Khattak et al., CVPR 2023). Base code: the official MaPLe repository
(`muzairkhattak/multimodal-prompt-learning`) and its dependency Dassl.pytorch, both pinned to a fixed
commit. The project tests MaPLe and related prompt-learning methods on DomainNet styles in the
paper's base-to-novel setting.

The work is done in separate sessions, one task at a time. The task for each session is given in the
prompt. Before writing code, read the existing code and reuse its structure, names, and helpers.

## How to work

- Do not guess. If a path, setting, value, or design choice is not given, ask.
- Never edit the base repositories. Put all new code in this project's own folders. If base-repo
  behavior must change, do it from our code and say so.
- Do not change hyperparameters, data splits, or test sets unless asked.
- Do not run long training jobs or submit cluster jobs unless asked.
- Report failed checks, errors, and surprising numbers exactly as they are. Never work around them
  silently.

## Requirements from the course (the code is graded on these)

- **README.md**: what the project does, how to install, how to get the data, and one command that
  reproduces the headline result.
- **Pinned dependencies** in `requirements.txt` or `environment.yml`.
- **CONTRIBUTIONS.md**: a table of components, their origin (base repo at its pinned commit, or new),
  and what was changed. Keep it accurate as code is added.
- **Smoke test**: one command that runs end to end in under 10 minutes on a single GPU on a tiny
  subset and produces output. Keep it working after every change.
- **Cluster job scripts** (GAIVI / Slurm) are part of the repository.
- **Seed control**: every run has a seed that is set, logged, and configurable.
- **Data handling**: a download script or exact instructions. Never commit datasets, or checkpoints
  or other files over 100 MB.
- **Run logging**: every run records its config, seed, git commit, GPU used, and result.
- **Structure**: configuration separated from logic, experiments runnable by script, results
  regenerable from code. No dead code, no unused notebooks in the root, names that say what they hold.
- **Reproducibility**: a clean clone must install and run the smoke test without edits. Every number
  in the report must be produced by the submitted code.

## Evaluation correctness (graded most strictly)

- No train/test leakage. Test images are never used for training, tuning, or picking checkpoints.
- No selection on test data: no early stopping or choosing prompts, templates, or epochs by test
  accuracy. Use the final epoch.
- Identical preprocessing and identical test images for every method being compared.
- Numbers come from code only. Never type a result into a table, CSV, figure, or document by hand.

## Code style

- Python 3.8 compatible (the base repo is tested on 3.8).
- I must be able to explain every line live, without notes. Keep code simple: small plain functions,
  clear names, no clever abstractions.
- Comment why a non-obvious choice was made, not what the line does.
- Use separate files for separate concerns. Avoid very long files.
- Use `pathlib`, `argparse`, and `logging`. Constants and settings live in one place, not scattered.
- Fail loudly with a message stating what was expected and what was found.
- Shell scripts start with `set -euo pipefail`.

## Tables and figures

- Tables: CSV (plus LaTeX if asked), numbers to 2 decimals, mean ± SD over seeds, number of seeds
  stated, anything with fewer than 3 seeds marked as preliminary.
- Figures: saved as files by code (PDF and PNG), labeled axes with units, error bars defined,
  readable in grayscale.

## Git

- Commit after each completed piece of work, small and focused, so the history shows the work
  developing over time. Message format: `<area>: <what changed>`.
- Never commit data, checkpoints, logs, or local config with paths or credentials.
- Do not rewrite history or force-push.
