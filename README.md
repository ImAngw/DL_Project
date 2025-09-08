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
  
  - `layers.py` module with the custom train function
</details>


