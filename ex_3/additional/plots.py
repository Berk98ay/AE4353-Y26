import matplotlib.pyplot as plt
import numpy as np
import torch


def _show(ax, img, img_shape):
    """Draw one flat face vector as an image on the given axis."""
    ax.imshow(np.asarray(img).reshape(img_shape), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def plot_preprocessing(dataset_file, n=6, img_shape=(32, 32)):
    """Show the raw FFHQ thumbnails next to the version the VAE actually sees.

    The top row is the original 128x128 color photograph, the bottom row is
    the same face after cropping, grayscaling and downsampling.

    Parameters:
    - dataset_file (str): path to FFHQ_FACES.npz.
    - n (int): number of faces to display.
    - img_shape (tuple): shape of the preprocessed faces.
    """
    from PIL import Image

    from additional.dataset import preprocess_face

    with np.load(dataset_file) as data:
        demo = data["demo_rgb"][:n]

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))

    for i in range(n):
        axes[0, i].imshow(demo[i])
        axes[0, i].axis("off")

        small = preprocess_face(Image.fromarray(demo[i]), size=img_shape[0])
        axes[1, i].imshow(small, cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original (128x128 RGB)", loc="left", fontsize=10)
    axes[1, 0].set_title(
        f"Preprocessed ({img_shape[0]}x{img_shape[1]} grayscale)", loc="left", fontsize=10
    )

    plt.suptitle("What the preprocessing does", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_reconstructions(model, data_loader, device, n=10, img_shape=(32, 32)):
    """Plot input faces above the reconstructions the VAE produces for them.

    Parameters:
    - model: trained VAE, returning (x_hat, mean, log_var).
    - data_loader: loader yielding (x, _) batches of flattened faces.
    - device: torch device to run inference on.
    - n (int): number of faces to display.
    - img_shape (tuple): shape to reshape each flat vector back into.
    """
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)

    with torch.no_grad():
        x_hat, _, _ = model(x)

    x = x.cpu()
    x_hat = x_hat.cpu()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for i in range(n):
        _show(axes[0, i], x[i], img_shape)
        _show(axes[1, i], x_hat[i], img_shape)

    axes[0, 0].set_title("Input", loc="left", fontsize=10)
    axes[1, 0].set_title("Reconstruction", loc="left", fontsize=10)

    plt.suptitle("Reconstructions", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_latent_samples(model, latent_dim, device, n=10, img_shape=(32, 32), seed=None):
    """Decode random points drawn from the prior to generate brand new faces.

    Nothing here is reconstructed from a real photograph: every image comes
    from a latent vector sampled straight out of N(0, I). How face-like these
    look tells you how well the KL term has organized the latent space.

    Parameters:
    - model: trained VAE, with a `.Decoder` attribute.
    - latent_dim (int): dimensionality of the latent space.
    - device: torch device to run inference on.
    - n (int): number of faces to generate.
    - img_shape (tuple): shape to reshape each flat vector back into.
    - seed (int): optional seed, so you can regenerate the same faces.
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        faces = model.Decoder(z).cpu()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.6))
    for i in range(n):
        _show(axes[i], faces[i], img_shape)

    plt.suptitle("Faces sampled from the prior", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_interpolation(faces, img_shape=(32, 32)):
    """Plot a walk through the latent space from one face to another.

    Parameters:
    - faces: tensor of decoded faces, ordered from start to end of the walk.
    - img_shape (tuple): shape to reshape each flat vector back into.
    """
    faces = faces.detach().cpu()
    n = len(faces)

    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.2))
    for i in range(n):
        _show(axes[i], faces[i], img_shape)

    axes[0].set_title("Face A", fontsize=10)
    axes[-1].set_title("Face B", fontsize=10)

    plt.suptitle("Interpolating in the latent space", fontsize=14)
    plt.tight_layout()
    plt.show()
