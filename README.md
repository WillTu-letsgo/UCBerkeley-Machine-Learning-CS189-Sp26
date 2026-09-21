# Machine Learning & Deep Learning — From Algorithms to Implementations

A collection of machine learning and deep learning projects developed through
UC Berkeley CS 189/289A coursework.

This repository focuses on the implementation side of machine learning:
turning mathematical ideas and model architectures into working code.

If you're interested in theoretical stuff (proofs, calculation, algorithm induction...), please refer to `HWs (ML Theory)`

---

## What This Repository Demonstrates

### 1. Implementing Machine Learning Algorithms

I worked with both classical and modern machine learning methods, including:

- K-means clustering
- Multilayer perceptrons
- Linear regression
- Data augmentation
- Rotation-invariant classification
- Test-time augmentation
- Model evaluation and error analysis

The emphasis is not only on training models, but also on understanding how
different modeling and data-processing choices affect model behavior.

---

### 2. Building Core ML Infrastructure from Scratch

One of the projects implements a minimal automatic differentiation framework
similar in spirit to PyTorch's computational graph.

I implemented:

- Tensor operations and computation graphs
- Automatic differentiation
- Topological sorting
- Reverse-mode backpropagation
- Gradient accumulation
- SGD
- Momentum
- Adam
- Muon with Newton–Schulz orthogonalization

This project helped me understand what happens underneath high-level
frameworks such as PyTorch.

---

### 3. Implementing Neural Network Architectures

I implemented several major neural network architectures directly in PyTorch,
including:

- Convolutional Neural Networks
- Residual Blocks
- ResNet-18
- Scaled Dot-Product Attention
- Multi-Head Attention
- Transformer Encoder
- Transformer Decoder
- Positional Encoding
- End-to-End Transformer

For the Transformer project, the implementation proceeds from the basic
attention operation to complete encoder/decoder architectures.

---

### 4. Working with Different Data Modalities

The projects cover more than conventional tabular or image data.

I worked with:

- Fashion-MNIST images
- Tabular regression data
- Natural language
- DNA sequences
- Audio signals
- Spectrograms
- Instruction-tuning datasets

Examples include:

- CNN/ResNet for image classification
- DNABERT for DNA sequence classification
- ConvNeXt for environmental sound classification
- Transformer-based language modeling
- Qwen2.5 fine-tuning for machine-learning question answering

This gave me experience adapting the same machine learning principles to
different forms of data rather than relying on a single application domain.

---

### 5. Transfer Learning and Modern Deep Learning

Later projects move from implementing models from scratch toward using and
adapting pretrained models.

Examples include:

- DNABERT with a custom classification head
- ConvNeXt pretrained on ImageNet
- Frozen-backbone vs. full fine-tuning
- Qwen2.5-0.5B-Instruct fine-tuning
- Hugging Face Transformers and TRL
- Supervised fine-tuning (SFT)
- Evaluation before and after fine-tuning

These 

---

## Project Overview

| Project | Main Topics | Key Implementations |
|---|---|---|
| **Project 1.1** | Classical ML & Data Analysis | K-means, MLP, confusion matrix, image transformations, tensor operations |
| **Project 1.2** | Regression & Robust Classification | Linear regression, rotation prediction, data augmentation, test-time augmentation |
| **Project 3** | Automatic Differentiation & Optimization | Computational graphs, backpropagation, SGD, Momentum, Adam, Muon |
| **Project 4.1** | Deep Learning Architectures | CNN, ResNet-18, Attention, Transformer |
| **Project 4.2** | Transfer Learning & Multimodal ML | DNABERT, ConvNeXt, spectrograms, fine-tuning |
| **Project 5** | LLM Fine-Tuning | Qwen2.5, Hugging Face, SFT, MMLU, evaluation pipeline |

---

## Technical Highlights

### Automatic Differentiation

Built a minimal autograd engine capable of constructing computational graphs
during the forward pass and propagating gradients through the graph during
backpropagation.

```text
Forward Pass
     ↓
Computational Graph
     ↓
Topological Ordering
     ↓
Reverse-Mode Autodiff
     ↓
Gradient Accumulation
     ↓
Optimizer