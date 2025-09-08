# Deep Learning Project
This repository contains a small-scale reproduction of the results reported in the paper:  
REFORMER: THE EFFICIENT TRANSFORMER - Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya  
https://arxiv.org/pdf/2001.04451

## Code organization

<details>
  <summary>📂 dl_proj/</summary>

  - `checkpoints/` contains the best models I trained
  - `configs/` includes all .yaml files used to configure the different experiments:
      - `cls_train.yaml` configuration for `cls_train.py`
      - `gen_train.yaml` configuration for `generation_train.py`
      - `perf_study.yaml` configuration for `performance_studies.py`
      - `synth_comparison.yaml` configuration for `synth_models_comparison.py`
      - `synth_train.yaml` configuration for `synth_train.py`
  - `models/` includes:
      - `classification_model.py`
      - `generation_model.py`
  - `synth_dataset_builder/` includes:
      - `synthetic_dataset.py` it generates the dataset used in `synth_train.py`
  - `utils/` includes:
      - `cls_train_utils.py` utility functions for `cls_train.py`
      - `generation_train_utils.py` utility functions for `generation_train.py`
      - `performance_utils.py` utility functions for `performance_studies.py`
      - `synth_train_utils.py` utility functions for `synth_train.py`
      - `graphic_utils.py` includes functions used to plot images and graphs
      - `positional_encoding.py`
</details>

<details>
  <summary>📂 my_custom_ai/</summary>  
  
  - `custom_train/` module with the custom train function
  - `utils` module with utilities
</details>

<details>
  <summary>📂 my_transformers/</summary>  
  
  - `layers.py`
  - `modules.py`
  - `local_sensitive_hashing.py`
  - `rev_layers.py`
  - `sublayers.py`
</details>

## Experiments

### Train a Classifier
In this experiment, a classifier was trained on MNIST/CIFAR-10. To replicate the training, run:
```bash
python3 cls_train.py --config dl_proj/configs/cls_train.yaml
```
By default, MNIST is used, but if you want to work with CIFAR-10, modify `dl_proj/configs/cls_train.yaml`.  
**Cross Entropy Loss** is used during the training (more details in `dl_proj/utils/cls_train_utils.py`).


### Train a Generator
In this experiment, an autoregressive model was trained to generate images (MNIST). For this task, scaled images ($dim=12×12$) were used during training to make the process faster. To replicate the training, run:

```bash
python3  generation_train.py --config dl_proj/configs/gen_train.yaml
```
If you want to change image dimensions modify `max_len` in `dl_proj/configs/gen_train.yaml`.  
**Focal Loss** is used in the training; **BPD Loss** is used in validation (more details in `dl_proj/utils/generation_train_utils.py`).  
You can also generate images using one of the trained models that you can find in `dl_proj/checkpoints/`. You can choose from:
* mnist-gen-full: model with full attention
* mnist-gen-lsh-r1-144: lsh attention, one hash round
* mnist-gen-lsh-r2-144: lsh attention, two hash rounds
* mnist-gen-lsh-r4-144: lsh attention, four hash rounds

To genarate images, run:
```bash
python3  mnist_generator.py --config dl_proj/configs/gen_train.yaml
```
By default, mnist-gen-lsh-r4-144 is used. To use a different model, change `experiment_name` in `dl_proj/configs/gen_train.yaml`.


### Performance study
In this experiment, we assess how well the algorithm employing LSH attention can reconstruct the attention matrix. We also record and compare the computation times and memory of full attention and LSH attention (for various numbers of rounds) when performing the qk dot product.  

To replicate the results, run:
```bash
python3 performance_studies.py --config dl_proj/configs/perf_study.yaml
```
See `dl_proj/configs/perf_study.yaml` for more details.


### Synthetic task:
This experiment involves training a model with LSH attention on a synthetic task. Given a set of sequences:  

$$
SOS, w_1, w_2, ..., w_n, SOS, w_1, w_2, ..., w_n 
$$

the objective is to train an autoregressive model that can accurately reproduce the input sequence.  
To replicate the training, run:
```bash
python3 synth_train.py --config dl_proj/configs/synth_train.yaml
```
See `dl_proj/configs/synth_train.yaml` for more details.  
The training was repeated for various numbers of rounds. To compare the inference accuracies, execute the following code:
```bash
python3 synth_models_comparison.py --config dl_proj/configs/synth_comparison.yaml
```
The following models are involved:
* synth-gen-full: full attention
* synth-gen-lsh-r1: lsh attention, one hash round
* synth-gen-lsh-r2: lsh attention, two hash rounds
* synth-gen-lsh-r4: lsh attention, four hash rounds
