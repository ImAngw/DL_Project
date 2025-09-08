import seaborn as sns
import matplotlib.pyplot as plt
import os


def draw_heatmap(matrix, directory=None, title = "Heat Map", annot=False, show=False):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", annot=annot)
    # plt.axis('off')
    plt.title(title)
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/{title}.pdf", format='pdf', bbox_inches='tight')
    if show:
        plt.show()

def plot_curves(curves, labels=None, show=False, directory=None,
                title="Curves", xlabel="x", ylabel="y", labels_x_axis=None, is_memory=False):

    plt.figure(figsize=(8, 5))

    for i, (x, y) in enumerate(curves):
        label = labels[i] if labels and i < len(labels) else f"Curve {i+1}"
        if label == 'Full attention':
            plt.plot(x, y, label=label, ls='--', color='black')
        else:
            plt.plot(x, y, label=label)

        if labels_x_axis is not None:
            plt.xscale('log')
            plt.xticks(x, labels_x_axis)
            plt.minorticks_off()
            plt.tick_params(axis='x', which='both', direction='out')

    if is_memory:
        plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, ls="--")
    plt.tight_layout()
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{directory}/{title}.pdf", format='pdf', bbox_inches='tight')
    if show:
        plt.show()


def show_image(img, label, vocab_size):
    channels = img.shape[0]

    img = img.detach().cpu().numpy()
    if img.shape[0] == 1:
        image_to_show = img.squeeze(0)  # (H, W)
    else:
        image_to_show = img.transpose(1, 2, 0)  # (H, W, C)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_show, cmap='gray' if channels == 1 else None, vmin=0, vmax=vocab_size if channels == 1 else vocab_size - 1)
    plt.axis('off')
    plt.title(f"Label: {label.item()}")
    plt.show()
