# [AE4353] Artificial Intelligence for Aerospace Control and Operations
> Welcome to the repository of the 2026/2027 [AE4353] Artificial Intelligence for Aerospace Control and Operations course! 🚀

## Table of Contents
- [About](#about)
- [Exercises](#exercises)
- [Environments](#environments)
    - [Which Should I Pick?](#which-should-i-pick)
    - [Getting Started](#getting-started)
- [Data](#data)
    - [Working on Kaggle](#working-on-kaggle)
    - [Working locally](#working-locally)
    - [Data Credits](#data-credits)
- [GitHub Copilot](#github-copilot)
- [Repository Layout](#repository-layout)
- [License](#license)
- [Contact Information](#contact-information)

## About
This repository contains the course materials for the exercises and the separate competition associated with the course. The exercise notebooks are kept in the repository, and the competition materials will be added separately as needed.

Exercise 0 is a warm-up that needs no data and no setup beyond the environment; start there if you want to refresh your Python before the real problems. The remaining exercises walk you through applying deep learning to aerospace control and perception problems: you will train a network to fly a quadrotor, build a solar compass from polarized images, and generate faces with a variational autoencoder. Each one is a Jupyter notebook with sections marked `TODO` for you to complete.

If you have any questions or need assistance, feel free to reach out — see [contact information](#contact-information). Happy coding and learning! 🌟

## Exercises

| # | Notebook | Topic | Data |
| --- | --- | --- | --- |
| 0 | [ex_0/ex_0.ipynb](ex_0/ex_0.ipynb) | **Python and Machine Learning Warm-Up** — Python and NumPy refresher, then a perceptron trained by hand with gradient descent | none (generated in the notebook) |
| 1 | [ex_1/ex_1.ipynb](ex_1/ex_1.ipynb) | **Quadrotor Flight with Deep Learning** — learn a control policy from simulated hover trajectories and evaluate it closed-loop | `2D_QUAD_HOVER.npz`, `3D_QUAD_HOVER.npz` |
| 2 | [ex_2/ex_2.ipynb](ex_2/ex_2.ipynb) | **Solar Compass from Polarization Images** — CNN regression of heading angle, with representation selection and data augmentation | `polarization_dataset/dataset.h5` |
| 3 | [ex_3/ex_3.ipynb](ex_3/ex_3.ipynb) | **Variational Autoencoders** — build and train a VAE on face images, and explore its latent space | `FFHQ_FACES.npz` |

Each exercise folder also contains a `*_kaggle.ipynb` variant. It is the **same exercise**, with file paths adapted to the Kaggle environment. Work through whichever one matches the environment you chose below — not both.

Supporting code lives in each exercise's `additional/` folder (plotting helpers, dataloaders, simulation code). We keep it out of the notebooks to keep them readable; you generally do not need to edit it, but you are welcome to read it.

## Environments
We officially support **two** environments this year. Both cover every exercise, and you can switch between them at any point.

### Which Should I Pick?

| | **Kaggle** ([kaggle.md](kaggle.md)) | **Local** ([local.md](local.md)) |
| --- | --- | --- |
| | *the go-to standard* | *recommended if you can manage it* |
| Runs on | Kaggle's servers, in your browser | Your own machine |
| Interface | Kaggle notebook editor | **VS Code — the same as the exam** |
| Setup effort | A Kaggle account and a per-exercise file upload | One-time install of VS Code, Git and Miniconda |
| Dependencies | Kaggle's pre-installed stack | Pinned by [env.yml](env.yml) — identical for everyone |
| GPU | Free GPU quota (~30 h/week) | Only if you own an NVIDIA GPU |
| Files | Each exercise's files must be uploaded as a Kaggle Dataset | Just clone the repo |
| Works offline | No | Yes |

**Kaggle is the standard, go-to environment for this course.** It works on any machine, needs nothing installed, and comes with a free GPU. If you are unsure where to start, start there — you will be running Exercise 1 within minutes.

**That said, we recommend the local setup to everyone who can manage it.** The reason is not speed: it is that the local setup runs in **VS Code, which is the same environment as the exam**. Getting comfortable with it now — the editor, selecting a kernel, navigating the file tree — is one less thing to work out on exam day. Kaggle's notebook editor, however convenient, is not what you will sit in front of during the exam.

So: **use Kaggle whenever you need it, and local whenever you can.** Switching between the two mid-course is completely fine.

> ⚠️ These two are what we support. You are of course free to use something else (Google Colab, your own cluster, a bare `pip` install), but we cannot help you debug it.

### Getting Started
- ☁️ **[kaggle.md](kaggle.md)** — Kaggle account, importing the notebooks, uploading datasets, and enabling the GPU. *Start here if you are unsure.*
- 💻 **[local.md](local.md)** — VS Code + Git + Miniconda, on Windows, Linux or macOS, including optional GPU setup. *Worth the effort — it is the exam environment.*

## Data
Exercise 0 needs no data at all — the notebook generates its own. Everything else is published as a **public Kaggle dataset**, and Exercises 1 and 2 are *additionally* in the SURFdrive download.

| Exercise | Kaggle dataset | Also in the SURFdrive download? |
| --- | --- | --- |
| 0 | — nothing to download | — |
| 1 | [`ae4353_1`](https://www.kaggle.com/datasets/quentinmissinne/ae4353-1) | yes |
| 2 | [`ae4353_2`](https://www.kaggle.com/datasets/quentinmissinne/ae4353-2) | yes |
| 3 | [`ae4353_3`](https://www.kaggle.com/datasets/quentinmissinne/ae4353-3) | no — Kaggle only |

### Working on Kaggle
There is nothing to download: you attach the dataset for your exercise as an *input* and the files appear under `/kaggle/input/`. The exact steps differ slightly per exercise — some datasets are ours to publish and you just attach them, others you upload once yourself. [kaggle.md](kaggle.md#4-per-exercise-instructions) has the list.

### Working locally
Two routes, and you can mix them freely:

- **SURFdrive** — this [link](https://surfdrive.surf.nl/files/index.php/s/uStySKYBKHBXcjP), password `Ae4353`. Extract it and place the resulting `AE4353-Datasets-2026` folder inside this repository's `data/` directory. This covers Exercises 1 and 2.
- **Kaggle** — open the dataset page for your exercise, press `Download`, unzip it, and move the data files into `data/AE4353-Datasets-2026/`. This is the only route for Exercise 3's `FFHQ_FACES.npz`, and it is a convenient alternative for Exercises 1 and 2 if you would rather not use SURFdrive.

Some of the Kaggle datasets also contain an `additional/` folder of helper scripts. You can ignore it — those scripts already come with this repository.

See [local.md](local.md#4-data) for the exact folder layout you should end up with.

> ⚠️ Please do not commit the dataset back to the repository.

### Data Credits
The faces used in Exercise 3 come from the [Flickr-Faces-HQ (FFHQ) dataset](https://github.com/NVlabs/ffhq-dataset) by Karras et al. (NVIDIA). The individual photographs were published on Flickr under Creative Commons (BY 2.0 or BY-NC 2.0) or public-domain terms; the collection as a whole is licensed by NVIDIA under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

We ship a **modified** subset: cropped, converted to grayscale and downsampled to 32×32. It carries the same CC BY-NC-SA 4.0 license, which permits non-commercial use with attribution — so please keep it to your coursework.

## GitHub Copilot
GitHub Copilot is an AI-powered assistant that helps you write code faster and more efficiently. It provides intelligent code suggestions and completions based on your context, enhancing your coding experience and boosting productivity.

> ⚠️ We encourage you to use tools like this to aid in learning concepts and practicing coding. However, it's important not to rely solely on these tools — ensure you put in the effort to understand and practice the material yourself! Please note that such tools will ***NOT*** be permitted during the final exam.

If you do not have it yet, please sign up for the Student Developer Pack on GitHub using this [link](https://education.github.com/pack). Once you have signed up, wait for GitHub to authenticate your request. Once authenticated, you will have access to GitHub Copilot, which you can enable in VS Code by installing the GitHub Copilot extension and signing in with your GitHub account.

## Repository Layout

```
AE4353-Y26/
├── README.md          # you are here — course and environment overview
├── local.md           # setup guide: VS Code + Miniconda on your machine
├── kaggle.md          # setup guide: running the exercises on Kaggle
├── env.yml            # conda environment definition (local setup)
├── data/              # place the downloaded dataset here (not committed)
├── ex_0/              # Exercise 0 — Python and ML warm-up
├── ex_1/              # Exercise 1 — quadrotor flight
├── ex_2/              # Exercise 2 — solar compass
└── ex_3/              # Exercise 3 — variational autoencoders
```

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.

> ⚠️ The MIT License covers the **code** in this repository. The course datasets are distributed separately under their own terms — see [Data Credits](#data-credits).

## Contact Information
For any questions or inquiries, please contact us at:

- Quentin Missinne: [Q.Missinne@tudelft.nl](mailto:Q.Missinne@tudelft.nl)
- Dequan Ou: [D.Ou@tudelft.nl](mailto:D.Ou@tudelft.nl)
- Reinier Vos: [R.W.Vos@tudelft.nl](mailto:R.W.Vos@tudelft.nl)

We will be happy to answer your questions and assist you! 🙂
