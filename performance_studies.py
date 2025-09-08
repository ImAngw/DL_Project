import torch
from tqdm import tqdm
import sys
import time
from dl_proj.utils.performance_utils import return_qkv, return_full_heatmap, return_lsh_heatmap, get_tensor_memory
from dl_proj.utils.graphic_utils import draw_heatmap, plot_curves
import yaml

from dl_proj.utils.synth_train_utils import MyConfig


def heatmap_approximation(configs):
    residuals = []
    # q shape (b_size, length, heads, dk); k shape (b_size, length, heads, dk); v shape (b_size, length, heads, dv);
    q, k, v = return_qkv(configs.b_size, configs.vocab_size, configs.length,
                         configs.embed_dim, configs.heads, configs.dk, configs.dv, configs.device)

    # full_heatmap shape: (length, length)
    full_heatmap = return_full_heatmap(q, q, v, configs.dk, configs.length, configs.device, configs.causal_mask)

    if configs.show_heatmaps:
        draw_heatmap(
            full_heatmap,
            directory=configs.directory if configs.save_heatmaps else None,
            title=f'Full attention',
            show=configs.show_heatmaps
        )

    q = q.reshape(configs.b_size, configs.length, configs.heads * configs.dk)
    # k = k.reshape(configs.b_size, configs.length, configs.heads * configs.dk)
    v = v.reshape(configs.b_size, configs.length, configs.heads * configs.dv)


    with tqdm(total=len(configs.n_rounds) * len(configs.bucket_sizes),
              desc=f'Work Done', file=sys.stdout, colour='green', ncols=100, dynamic_ncols=True) as pbar:

        for bucket_size in configs.bucket_sizes:
            residuals_per_bucket = []

            for round in configs.n_rounds:
                lsh_heatmap = return_lsh_heatmap(
                    q, q, v, configs.dk, configs.dv, configs.heads,
                    configs.length, round, bucket_size, configs.device, configs.causal_mask
                )

                if configs.show_heatmaps:
                    draw_heatmap(
                        lsh_heatmap,
                        directory=configs.directory if configs.save_heatmaps else None,
                        title=f'LSH attention (n_round {round})',
                        show=configs.show_heatmaps
                    )

                se = ((lsh_heatmap - full_heatmap) ** 2).mean()
                error = se ** 0.5

                '''delta = torch.abs(lsh_heatmap - full_heatmap).sum()
                full = full_heatmap.sum()
                error = delta / full'''

                residuals_per_bucket.append(error.item())
                pbar.update(1)
            residuals.append([configs.n_rounds, residuals_per_bucket])

    plot_curves(
        curves=residuals,
        labels=[f'bucket size = {bucket_size}' for bucket_size in configs.bucket_sizes],
        show=configs.show_graphics,
        directory=configs.directory if configs.save_graphics else None,
        title=f'LSH Attn Reconstruction (length={configs.length})',
        xlabel='N rounds', ylabel='RMSE'
    )

def time_and_memory_monitoring(configs):
    n_rounds = [1, 2, 4, 8]
    lengths = [2048, 4096, 8192, 16384]
    b_sizes = [8, 4, 2, 1]
    bucket_size = 64

    full_times = []
    lsh_times = [[] for _ in range(len(n_rounds))]
    times_for_plot = []
    labels = []
    labels_x_axis = []


    full_memory = []
    lsh_memory = [[] for _ in range(len(n_rounds))]
    memory_for_plot = []
    labels_m = []
    labels_x_memory = []


    for i, length in enumerate(lengths):
        label = str(length) + '/' + str(b_sizes[i])
        labels_x_axis.append(label)
        labels_x_memory.append(str(length))


    for i, length in enumerate(lengths):
        q = torch.randn(b_sizes[i], length, configs.heads, configs.dk, device=configs.device)
        k = torch.randn(b_sizes[i], length, configs.heads, configs.dk, device=configs.device)

        start = time.time()
        attn = torch.einsum('bqhd, bkhd -> bhqk', [q, k])
        stop = time.time()
        full_times.append(stop - start)

        # Memory monitoring
        memory = get_tensor_memory(attn)
        full_memory.append(memory)

        for j, n_round in enumerate(n_rounds):
            q = torch.randn(n_round, b_sizes[i], length // bucket_size, 2 * bucket_size, configs.heads, configs.dk, device=configs.device)
            k = torch.randn(n_round, b_sizes[i], length // bucket_size, 2 * bucket_size, configs.heads, configs.dk, device=configs.device)
            start = time.time()
            attn = torch.einsum('rbnqhd, rbnkhd -> rbnhqk', [q, k])
            stop = time.time()

            lsh_times[j].append(stop - start)

            memory = get_tensor_memory(attn)
            lsh_memory[j].append(memory)


    times_for_plot.append([lengths, full_times])
    labels.append('Full attention')

    memory_for_plot.append([lengths, full_memory])
    labels_m.append('Full attention')

    for i, results in enumerate(lsh_times):
        times_for_plot.append([lengths, results])
        labels.append(f'LSH {i + 1} rounds')

        memory_for_plot.append([lengths, lsh_times[i]])
        labels_m.append(f'LSH {i + 1} rounds')


    plot_curves(
        curves=times_for_plot,
        labels=labels,
        show=True,
        directory=configs.directory,
        title=f'Time VS Length',
        xlabel='Length', ylabel='Time [s]',
        labels_x_axis=labels_x_axis,
    )

    plot_curves(
        curves=memory_for_plot,
        labels=labels_m,
        show=True,
        directory=configs.directory,
        title=f'Memory VS Length',
        xlabel='Length', ylabel='Memory [MB]',
        labels_x_axis=labels_x_memory,
        is_memory=True
    )





if __name__ == '__main__':
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = SimpleNamespace(**cfg_dict)


    torch.manual_seed(config.seed)
    heatmap_approximation(config)
    time_and_memory_monitoring(config)

    # python3 performance_studies.py --config dl_proj/configs/perf_study.yaml

