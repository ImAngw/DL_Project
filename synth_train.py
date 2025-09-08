from dl_proj.synth_dataset_builder.synthetic_dataset import SyntheticDataset
from torch.utils.data import DataLoader
from dl_proj.utils.synth_train_utils import MyConfig, SyntFunctionContainer, get_model
from my_custom_ai.custom_train.train import CustomTraining
import torch



def main(configs):
    train = SyntheticDataset(
        seq_len=configs.max_len//2,
        n_samples=configs.n_samples,
        vocab_size=configs.vocab_size,
    )

    train_loader = DataLoader(train, batch_size=configs.batch_size, shuffle=True)

    validation = SyntheticDataset(
        seq_len=configs.max_len//2,
        n_samples=configs.val_samples,
        vocab_size=configs.vocab_size,
    )

    val_loader = DataLoader(validation, batch_size=configs.batch_size, shuffle=False)

    model = get_model(configs)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    function_container = SyntFunctionContainer(configs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-3,
        steps_per_epoch=len(train_loader),
        epochs=configs.num_epochs,
        anneal_strategy='cos',
        final_div_factor=1e4,
        div_factor=50,
        pct_start=0.25
    )

    custom_train = CustomTraining(
        configs=configs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        function_container=function_container,
        scheduler=scheduler,
        step_scheduler_each_batch=True,
        eval_on_validation=True
    )

    custom_train.train()


if __name__ == "__main__":
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

    # python3 synth_train.py --config dl_proj/configs/synth_train.yaml
