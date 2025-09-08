from dl_proj.utils.generation_train_utils import  get_gen_model
from dl_proj.utils.synth_train_utils import MyConfig

import torch
from tqdm import tqdm
import sys
from dl_proj.utils.graphic_utils import show_image



def generation(configs, n_samples, top_k=1):

    dim = int(configs.max_len ** 0.5)

    model_dir = f"dl_proj/checkpoints/{configs.exp_name}.pth"

    model = get_gen_model(configs=configs).to(configs.device)
    model.load_state_dict(torch.load(model_dir))

    sos = configs.vocab_size

    model.eval()
    with torch.no_grad():
        for label in configs.labels:
            label = torch.tensor([label], device=configs.device, dtype=torch.int).unsqueeze(0)
            for i in range(n_samples):
                gen_img = torch.tensor([sos], device=configs.device, dtype=torch.int).unsqueeze(0)
                with tqdm(total=configs.max_len, desc=f'GENERATION DIGIT {label.item()}', file=sys.stdout, colour='blue',
                          ncols=100, dynamic_ncols=False) as pbar:
                    for j in range(configs.max_len):
                        output = model(x=gen_img, label=label)

                        probs = torch.softmax(output, dim=-1)
                        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                        last_token_top_k = top_k_probs[:, -1, :]
                        last_idx = top_k_indices[:, -1, :].squeeze(0)
                        sampled_idx = torch.multinomial(last_token_top_k, 1).squeeze(-1)
                        prediction = last_idx[sampled_idx.item()].unsqueeze(0).unsqueeze(0)
                        gen_img = torch.cat((gen_img, prediction), dim=1)

                        pbar.update(1)

                gen_img = gen_img[:, 1:].reshape(1, dim, dim)
                show_image(gen_img, label.detach().clone(), configs.vocab_size + 2)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict['save_on_wb'] = False
    config = MyConfig(**cfg_dict)
    config.task = 'gen'
    config.dataset = 'mnist'
    config.causal_mask = False
    config.self_attn_mask = False


    generation(config, n_samples=1)
    # python3  mnist_generator.py --config dl_proj/configs/gen_train.yaml
