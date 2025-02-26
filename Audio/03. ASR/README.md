# First try to create project with template

It's first try to create project with Template. I need to study this template!=)

# About this project

This is [first homework](https://github.com/markovka17/dla/tree/2024/hw1_asr) on course [DL in Audio from HSE](https://github.com/markovka17/dla).

In this project i must to implement and train a neural-network speech recognition system with a CTC loss. I cannot use implementations available on the internet! I need to learn and write one of the architecture ASR.

But i think, i can use some after i will write my architecture!)


# Recommended workflow

Recommended architectures:
- [DeepSpeech2](https://proceedings.mlr.press/v48/amodei16.pdf);
- [QuartzNet](https://arxiv.org/pdf/1910.10261); Note: it is difficult to train without a large batch size and nice GPU.
- [Jasper](https://arxiv.org/pdf/1904.03288); Note: it is difficult to train without a large batch size and nice GPU.
- [Conformer](https://arxiv.org/pdf/2005.08100);
- [wav2vec](https://arxiv.org/pdf/2006.11477); Note: it is difficult to train without a large batch size and nice GPU.
- [Whisper](https://arxiv.org/pdf/2212.04356) Note: it is difficult to train without a large batch size and nice GPU.

# Choice of architecture
My PC have:
- CPU: Ryzen 7 2700x
- GPU: Nvidia GTX 1070 8Gb
- RAM: 32 Gb

**DeepSpeech2**
Pros: Simple RNN-based model, easier to implement, requires less compute power.
Cons: Slower inference due to RNNs, training can be slow.
Best for: Your setup since it's the simplest architecture to implement and train.

**QuartzNet / Jasper**
Pros: CNN-based models, efficient inference.
Cons: Requires large batch sizes and longer training times.
Best for: If you can optimize batch sizes and use mixed precision training.

**Conformer**
Pros: Best accuracy, combines CNN + Transformer, state-of-the-art.
Cons: Very expensive to train, Transformers need high memory.
Not recommended for local training.

I will realize [DeepSpeech2](https://proceedings.mlr.press/v48/amodei16.pdf) on Pytorch.

# About DeepSpeech 2:
- [Paper Deep Speech 2](https://proceedings.mlr.press/v48/amodei16.pdf)
- [Deep Speech 2 | Lecture 75 (Part 3)](https://www.youtube.com/watch?v=OAJNSJSQn-w&ab_channel=MaziarRaissi)
- []()
# Data
I think, first i will use ASR only in english language.

- **Common Voice (Mozilla)** ([https://commonvoice.mozilla.org/ru/datasets](https://commonvoice.mozilla.org/ru/datasets))
- **Open STT** ([https://github.com/snakers4/open_stt](https://github.com/snakers4/open_stt))
- **Multilingual LibriSpeech (MLS)** ([https://www.openslr.org/94/](https://www.openslr.org/94/))
- **VoxForge** ([http://www.voxforge.org/ru](http://www.voxforge.org/ru))

## Optional
 **Cleaning and preprocessing**:
    - Noise removal, short/long recordings filtering.
    - Text normalization (bringing to a common format).
    - Split into training, validation and test samples.


# Implementations available on the internet
## HuggingFace 
- [HuggingFace course & tutorial](https://huggingface.co/learn/audio-course/chapter5/introduction)
- [ASR models on HuggingFace](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)


## Nvidia models:
### Automatic Speech Recognition
- [ASR](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/intro.html#:~:text=24.07%20documentation.-,Automatic%20Speech%20Recognition%20(ASR),-%23)
### On Russian:
- [STT Ru Quartznet15x5](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_ru_quartznet15x5)
- [STT RU FastConformer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_ru_fastconformer_hybrid_large_pc)

# This is template of project
## Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

### About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

### Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

### How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

### Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

### License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
