from dl_proj.utils.cls_train_utils import ClassificationContainer, get_cls_model, get_loaders
from my_custom_ai.custom_train.train import CustomTraining
import torch
from dl_proj.utils.synth_train_utils import MyConfig
import yaml


def cls_train_main(configs):
    train_loader, test_loader, n_patches, patch_dim = get_loaders(configs)
    print(f'- NUM PATCHES (SEQ LEN): {n_patches}      PATCH DIM: {patch_dim}')


    container = ClassificationContainer(configs=configs)
    model = get_cls_model(configs=configs, n_patches=n_patches, patch_dim=patch_dim)

    n_params = 0
    for param in model.parameters():
        n_params += param.flatten().shape[0]


    print(f'- DATASET: {configs.dataset.title()}    LENGTH: {len(train_loader) * configs.batch_size}     PIXELs PER IMG: {configs.max_len}')
    print(f'- TASK: Classification')
    print('- PARAMETERS: {}'.format(n_params))
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(train_loader),
        epochs=configs.num_epochs,
        anneal_strategy='cos',
        final_div_factor=1e4,
        div_factor=50,
        pct_start=0.3
    )

    custom_train = CustomTraining(
        configs=configs,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        function_container=container,
        scheduler=scheduler,
        step_scheduler_each_batch=True,
        eval_on_validation=True,
        grad_accum_steps=1
    )

    custom_train.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = MyConfig(**cfg_dict)
    config.task = 'cls'
    cls_train_main(config)


    # python3 cls_train.py --config dl_proj/configs/cls_train.yaml