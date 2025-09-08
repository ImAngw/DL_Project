# Deep Learning Project
This repository contains a small-scale reproduction of the results reported in the paper:  
REFORMER: THE EFFICIENT TRANSFORMER - Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya  
https://arxiv.org/pdf/2001.04451

<details>
  <summary>📂 Code organization</summary>

  - `dl_proj/checkpoints/` contains the best models I trained
  - `dl_proj/configs/` includes all .yaml files used to configure the different experiments:
      - `cls_train.yaml` configuration for `cls_train.py`
      - `gen_train.yaml` configuration for `generation_train.py`
      - `perf_study.yaml` configuration for `performance_studies.py`
      - `synth_comparison.yaml` configuration for `synth_models_comparison.py`
      - `synth_train.yaml` configuration for `synth_train.py`
  - `dl_proj/models/` includes:
      - `classification_model.py`
      - `generation_model.py`

</details>



