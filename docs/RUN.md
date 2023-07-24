# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each prompting variant including MaPLe, vision, language and independent V-L prompting.
Make sure to configure the dataset paths in environment variable `DATA` and run the commands from the main directory `multimodal-prompt-learning/`.
Below we provide training and evaluation instructions for MaPLe. The same instructions applies for all other variants including *Vision (VPT), Language and independent V-L prompting*.


### Training time and compute
We train MaPLe on each dataset with a batch size of 4 using a **single** NVIDIA A100 GPU.
Training MaPle on ImageNet for 5 epochs takes 1 hour for a single seed. So results for 3 seeds takes around 3 hours. For all remaining 10 datasets, it combinedly takes around 4 hours (for all 3 seeds) on a single A100 GPU. To ease reproduction of MaPLe results, we have provided [training logs](https://drive.google.com/drive/folders/1EvuvgR8566bL0T7ucvAL3LFVwuUPMRas?usp=sharing) for all datasets. 

## MaPLe

#### (1) Base-to-Novel class generalization setting
The default training settings are provided in config file at `configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml`. All hyper-parameters such as prompt length, prompt depth, etc., can be modified using this config file.

Below, we provide instructions to train MaPLe on imagenet. 


```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh imagenet 1
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 1

# seed=2
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh imagenet 2
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 2

# seed=3
# trains and evaluates on base classes
bash scripts/maple/base2new_train_maple.sh imagenet 3
# evaluates on novel classes
bash scripts/maple/base2new_test_maple.sh imagenet 3
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– MaPLe/
|   |   |   |   |   |–– vit_b16_c2_ep5_batch4_2ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx --test-log
```

The above steps can be repeated for other individual datasets.

#### Reproducing results using pre-trained weights for base-to-novel generalization setting

We show an example to reproduce results for imagenet. Follow the instructions below to reproduce results using our pre-trained model weights:
* Download the zipped folder containing pre-trained weights for a single dataset from this [link](https://drive.google.com/drive/folders/1-tB6BUDBzs9CXTOJ7p5hM4Svq1tL_mGz?usp=sharing). Additionally we also provide the log files for both training and evaluation. After unzipping, the directory should look like this:

```
imagenet
|–– base/
|   |–– seed1/
|   |–– seed2/
|   |–– seed3/
|–– novel/
|   |–– seed1/
|   |–– seed2/
|   |–– seed3/
```

Now use the evaluation script `scripts/maple/reproduce_maple.sh` and run the commands below to calculate the averaged results:
```bash
# evaluate on base and novel classes for SEED1
bash scripts/maple/reproduce_maple.sh imagenet 1 /path/to/imagenet/weights/folder
# evaluate on base and novel classes for SEED2
bash scripts/maple/reproduce_maple.sh imagenet 2 /path/to/imagenet/weights/folder
# evaluate on base and novel classes for SEED3
bash scripts/maple/reproduce_maple.sh imagenet 3 /path/to/imagenet/weights/folder
```

This should evaluate and save the log files in `output/` directory. To obtain the averaged results, run:

```bash
# prints averaged results for base classes
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx
# averaged results for novel classes
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx --test-log
```


#### (2) Cross-Dataset Transfer
We provide instructions to train MaPLe on imageNet using all 1000 classes and then evaluating it directly on new downstream datasets.
We provide cross-dataset config for MaPLe: `configs/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml`.
* Firstly, train MaPLe on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/maple/xd_train_maple.sh imagenet 1
# seed=2 
bash scripts/maple/xd_train_maple.sh imagenet 2
# seed=3 
bash scripts/maple/xd_train_maple.sh imagenet 3
```

* Now evaluate imageNet model on downstream datasets.

```bash
for SEED in 1 2 3
do
    bash scripts/maple/xd_test_maple.sh caltech101 ${SEED}
    bash scripts/maple/xd_test_maple.sh oxford_pets ${SEED}
    bash scripts/maple/xd_test_maple.sh stanford_cars ${SEED}
done
```

#### (3) Domain Generalization 
We use imagenet trained MaPLe model for domain generalization experiments. The steps are similar to above cross-dataset experiments, however, model is evaluated on imagenet variants.
* Evaluate imageNet model on variants of imagenet (domain shift datasets).

```bash
for SEED in 1 2 3
do
    bash scripts/maple/xd_test_maple.sh imagenetv2 ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet_sketch ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet_a ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet_r ${SEED}
done
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.
<br>


#### Reproducing official results for cross-dataset and domain generalization setting

We provide the instructions below to reproduce domain-generalization and cross-datasets results using our pre-trained imagenet model weights for MaPLe:
* Download the zipped folder containing pre-trained weights for imagenet from this [link](https://drive.google.com/drive/folders/1bmhvmNZc13WJ5U71qt0t8k91wyuoemVF?usp=sharing). Additionally, we also provide the log files for both training and evaluation. After unzipping, the directory should look like this:

```
imagenet
|–– seed1/
|–– seed2/
|–– seed3/
```

Now use the evaluation script `scripts/maple/reproduce_maple_xd.sh` and run the commands below to calculate the averaged results:
```bash
# evaluate on given dataset for SEED1
bash scripts/maple/reproduce_maple_xd.sh food101 1 /path/to/imagenet/weights/folder
# evaluate on given dataset for SEED2
bash scripts/maple/reproduce_maple_xd.sh food101 2 /path/to/imagenet/weights/folder
# evaluate on given dataset for SEED3
bash scripts/maple/reproduce_maple_xd.sh food101 3 /path/to/imagenet/weights/folder
```

This should evaluate and save the log files in `output/` directory. To obtain the averaged results, run:

```bash
# prints averaged results for food101 dataset
python parse_test_res.py output/evaluation/MaPLe/vit_b16_c2_ep5_batch4_2ctx_cross_datasets_16shots/food101 --test-log
```


#### Training and Evaluating other variants

For other variants including vision, language and independent V-L prompting techniques, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– MaPLe/
|   |–– IVLP/
|   |–– VPT/
```

```
scripts
|–– cocoop/
|–– coop/
|–– language-prompting/
|–– maple/
|–– independent-vlp/
```

Please use the corresponding config and script files and follow the same instructions as provided for MaPLe in order to train and evaluate the other variants. Same instructions can be followed to reproduce results of other variants using provided pretrained weights.
This repository also supports using official [CoOp](CoOp.md) and [Co-CoOp](Co-CoOp.md) configs and models.
