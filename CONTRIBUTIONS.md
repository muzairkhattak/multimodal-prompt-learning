Base: github.com/muzairkhattak/multimodal-prompt-learning @ 69bce21ae8eda80ad6187534b2dce09cf6c59e17

Dependency: github.com/KaiyangZhou/Dassl.pytorch @ c61a1b570ac6333bd50fb5ae06aea59002fb20bb

| Component | Origin | What I changed |
| --- | --- | --- |
| Training loop and trainers (MaPLe, IVLP, CoCoOp) | Base repo @ 69bce21 | Unmodified |
| `.gitignore` | New | Keeps outputs, logs, datasets and checkpoints out of git |
| `tools/data/common.sh` | New | Download, checksum and image count helpers shared by the download scripts |
| `tools/data/download_domainnet.sh` | New | Downloads the 6 DomainNet styles and the cleaned train/test lists, prints image counts |
| `tools/data/download_eurosat_dtd_aircraft.sh` | New | Downloads EuroSAT, DTD and FGVC Aircraft into the `docs/DATASETS.md` layout; EuroSAT comes from the authors' Zenodo record because the documented host timed out |
| `tools/check_env.py` | New | Prints torch and CUDA versions, runs CLIP ViT-B/16 once on the GPU |
| `slurm/test_env.sbatch` | New | One GPU, 10 minute GAIVI job that runs the environment check |
| `requirements.txt` | Base repo @ 69bce21 | Replaced its 3 lines with every pip-installed version in `environment_freeze.txt`, plus `wheel`, which `tb-nightly` requires |
| `environment_freeze.txt` | New | `conda list` export of the environment on GAIVI |
| `README.md` | New | Project summary, pinned commits and GAIVI setup. The base repo's README is kept at `docs/ORIGINAL_README.md` |
| `tools/data/write_domainnet_notes.py` | New | Generates `data/README.md` from the DomainNet lists |
| `data/README.md` | New | Generated notes: cleaned-list choice, fixed base/new split, classes missing from painting, classes with fewer than 16 training images |
