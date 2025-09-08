from dl_proj.utils.synth_train_utils import MyConfig
from dl_proj.utils.cls_train_utils import get_loaders
from dl_proj.utils.generation_train_utils import GenerationContainer, get_gen_model
from my_custom_ai.custom_train.train import CustomTraining
import torch



def return_pixel_recurrency(loader, vocab_size, n_images):
    counts = torch.zeros(vocab_size + 1, dtype=torch.long)
    for image, _, _ in loader:
        mgs = image.to(torch.int)
        counts += torch.bincount(mgs.view(-1), minlength=vocab_size + 1)
    counts[counts == 0] = 1
    counts = torch.cat((counts, torch.tensor([n_images])), dim=0)
    return counts

def train_main(configs):
    train_loader, test_loader, _, _ = get_loaders(configs)
    counts = return_pixel_recurrency(train_loader, configs.vocab_size, len(train_loader.dataset))
    container = GenerationContainer(configs=configs, pixel_recurrency=counts)
    model = get_gen_model(configs=configs).to(configs.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
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
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = MyConfig(**cfg_dict)
    config.task = 'gen'
    config.dataset = 'mnist'

    train_main(config)

    #  python3  generation_train.py --config dl_proj/configs/gen_train.yaml