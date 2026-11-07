# Prompt Learning on DomainNet Styles

Graduate class project for CIS 6930 Generative AI at USF. It tests MaPLe (Multi-modal Prompt
Learning, Khattak et al., CVPR 2023) and related prompt-learning methods on the DomainNet styles,
in the paper's base-to-novel setting: train on the base classes, then evaluate on the new classes.

Base code: [muzairkhattak/multimodal-prompt-learning](https://github.com/muzairkhattak/multimodal-prompt-learning)
at commit [`69bce21`](https://github.com/muzairkhattak/multimodal-prompt-learning/tree/69bce21ae8eda80ad6187534b2dce09cf6c59e17),
with [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) at commit
[`c61a1b5`](https://github.com/KaiyangZhou/Dassl.pytorch/tree/c61a1b570ac6333bd50fb5ae06aea59002fb20bb).
The original README is at [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md). See
[CONTRIBUTIONS.md](CONTRIBUTIONS.md) for what is new in this repository.

## Setup on GAIVI

Run every command from the repository root.

**1. Clone**

```bash
git clone https://github.com/HeyManan/multimodal-prompt-learning.git
cd multimodal-prompt-learning
```

**2. Create the environment**

```bash
conda create -y -n maple python=3.8.20 pip=24.2
conda activate maple
pip install -r requirements.txt

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git ../Dassl.pytorch
git -C ../Dassl.pytorch checkout c61a1b570ac6333bd50fb5ae06aea59002fb20bb
(cd ../Dassl.pytorch && python setup.py develop)
```

**3. Set `$DATA` and `$OUTPUT`**

`$DATA` holds the datasets and `$OUTPUT` holds the run outputs. Each GAIVI storage area has a 100 GB
quota per user, and the four datasets below need about 25 GB. `/general` is GAIVI's hot data area.

```bash
export DATA=/general/$USER/data
export OUTPUT=/general/$USER/output
mkdir -p "$DATA" "$OUTPUT"
```

**4. Download the CLIP weights**

This caches ViT-B/16 in `~/.cache/clip`. Run it on the login node, because the GAIVI docs do not say
whether compute nodes can reach the internet.

```bash
python -c "from clip import clip; clip._download(clip._MODELS['ViT-B/16'])"
```

**5. Check the environment**

```bash
sbatch slurm/test_env.sbatch
cat test_env_<job id>.out
```

The job uses the `ClsGenAI` partition, so change that line on another cluster. It passes when the
last line of the output is `INFO: Environment check passed.`

**6. Download the datasets**

Both scripts read `$DATA`, resume after an interruption, and stop with an error if the result is
incomplete. `nohup` keeps them running after you log out.

```bash
nohup bash tools/data/download_domainnet.sh > download_domainnet.log 2>&1 &
nohup bash tools/data/download_eurosat_dtd_aircraft.sh > download_eurosat_dtd_aircraft.log 2>&1 &
```

`download_domainnet.log` ends with a table of image counts per style:

- `on_disk` is larger than `train` + `test`, because the zips also hold the images that the cleaned
  lists removed.
- `missing` must be 0 for every style.

The cleaned-list choice, the classes missing from the painting lists, and the classes with few
training images are recorded in [data/README.md](data/README.md).
