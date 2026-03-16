# Classifier-Free Guidance in Diffusion Models

This repository contains the implementation of classifier-free guidance in diffusion models, a technique that allows for improved sample quality without the need for an external classifier. The code is based on tsmatz's implementation of classifier-free guidance in diffusion models, which can be found [here](https://github.com/tsmatz/diffusion-tutorials/blob/master/07-classifier-free-guidance.ipynb).

## Table of Contents

- [Overview](#overview)
    - [Dataset Considered](#dataset-considered)
    - [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

The purpose of this repository is to provide a clear and concise implementation of classifier-free guidance in diffusion models.

### Dataset Considered

The data used in this implementation is the MNIST dataset, which consists of handwritten digits, of which there are 10 classes (0-9). The implementation includes the training of a diffusion model on the MNIST dataset, as well as the application of classifier-free guidance during sampling to improve the quality of generated samples.

### Model Architecture

The model architecture used in this implementation is a simple U-Net architecture, which is commonly used in diffusion models. The U-Net consists of an encoder and a decoder, with skip connections between corresponding layers in the encoder and decoder. The model takes as input a noisy image, a timestep, and a class label (which can be set to a special value for unconditioned sampling), and outputs a denoised image.

<ins>Note:</ins> The timestep and class labels are embedded into a higher-dimensional space before being fed into the model (through the ResNet blocks), which allows the model to learn the relationship between the noise level, class information, and the image content to extract (through "conditioning" of the kernels/channels).

## Installation

To install the necessary dependencies for this project, you can use the following command:

```bash
pip install -r requirements.txt
```

## Usage

The main script for training and sampling from the diffusion model is `main.py` (based on the user's choice/input).

## Notes

The final saved models + embeddings are located in the `src/models/saved/` folder. 