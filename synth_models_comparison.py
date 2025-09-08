from dl_proj.utils.synth_train_utils import MyConfig, get_model, SyntFunctionContainer
from dl_proj.synth_dataset_builder.synthetic_dataset import SyntheticDataset
from torch.utils.data import DataLoader
import torch
from tabulate import tabulate



def full_score(configs, model_name, test_loader):
    model_dir = f"dl_proj/checkpoints/{model_name}.pth"
    configs.attn_type = 'full'
    container = SyntFunctionContainer(configs=configs)
    model = get_model(configs)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    score = container.validation_performance(model, test_loader)['score']
    return score

def lsh_scores(configs, model_name, test_loader):
    model_dir = f"dl_proj/checkpoints/{model_name}.pth"
    configs.attn_type = 'lsh'
    container = SyntFunctionContainer(configs=configs)

    allowed_rounds = [1, 2, 4]
    all_scores = []

    for n_rounds in allowed_rounds:
        configs.n_rounds = n_rounds
        model = get_model(configs)
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        score = container.validation_performance(model, test_loader)['score']
        all_scores.append(score)

    return all_scores

def print_table(values):
    headers = ['LSH Model', 'Train Rounds',  'Test Score (1 round)', 'Test Score (2 round)', 'Test Score (4 round)']
    print(tabulate(values, headers=headers, tablefmt="grid"))

def main(configs):
    lsh_model_names = {'synth-gen-lsh-r1':1, 'synth-gen-lsh-r2':2, 'synth-gen-lsh-r4':4}
    full_model_name = 'synth-gen-full'

    h_scores = []
    test_set = SyntheticDataset(
        seq_len=configs.max_len // 2,
        n_samples=configs.val_samples,
        vocab_size=configs.vocab_size,
    )
    test_loader = DataLoader(test_set, batch_size=configs.batch_size, shuffle=False)

    for model_name, train_rounds in lsh_model_names.items():
        score = lsh_scores(configs=configs, model_name=model_name, test_loader=test_loader)
        h_scores.append([model_name, train_rounds] + score)

    print_table(h_scores)

    f_score = full_score(configs=configs, model_name=full_model_name, test_loader=test_loader)
    print()
    print(f"{full_model_name} --> Full attn score: {f_score}")



if __name__ == '__main__':

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = MyConfig(**cfg_dict)
    main(config)

    # python3 synth_models_comparison.py --config dl_proj/configs/synth_comparison.yaml
