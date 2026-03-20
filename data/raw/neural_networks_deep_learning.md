# Neural Networks and Deep Learning

## What is a Neural Network?

A neural network is a computational model inspired by the structure of the brain. It consists of layers of interconnected nodes ("neurons") that learn to perform tasks by processing examples.

The breakthrough insight: instead of hand-crafting features, let the network learn its own representations from raw data through backpropagation and gradient descent.

---

## Anatomy of a Neural Network

### Neuron (Perceptron)
```
output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias)
```
- **Inputs (x)**: features or outputs from previous layer
- **Weights (w)**: learned parameters — how much each input matters
- **Bias**: shifts the activation function
- **Activation**: non-linear transformation applied to the weighted sum

### Layers
- **Input layer**: raw features
- **Hidden layers**: intermediate representations
- **Output layer**: final prediction (probabilities, values, etc.)

**Fully connected (Dense) layer**: every neuron connects to every neuron in the next layer.

### Activation Functions

| Function | Formula | Use Case |
|---|---|---|
| ReLU | max(0, x) | Hidden layers — fast, sparse, default choice |
| Leaky ReLU | max(0.01x, x) | Prevents "dying ReLU" problem |
| GeLU | x × Φ(x) | Transformers (BERT, GPT) |
| SiGLU / SwiGLU | x × σ(x) | LLaMA, PaLM — better than GeLU in practice |
| Sigmoid | 1/(1+e⁻ˣ) | Binary output (0 to 1) |
| Softmax | eˣᵢ / Σeˣⱼ | Multi-class output (probabilities sum to 1) |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs, outputs in (-1, 1) |

---

## Training: Forward Pass and Backpropagation

### Forward Pass
Input → Layer 1 → Layer 2 → ... → Output → Loss

### Loss Functions
- **MSE** (regression): `L = (y_pred - y_true)²`
- **Cross-Entropy** (classification): `L = -Σ y_true × log(y_pred)`
- **Binary Cross-Entropy**: single-class version of cross-entropy

### Backpropagation
Use the chain rule to compute the gradient of the loss with respect to every weight in the network:
```
∂L/∂w = ∂L/∂output × ∂output/∂w
```

Gradients flow backward through the network, updating each weight to reduce the loss.

### Gradient Descent
Update weights in the direction that reduces loss:
```
w = w - learning_rate × ∂L/∂w
```

**Variants:**
- **SGD**: one sample at a time — noisy, slow
- **Mini-batch SGD**: batches of 32–256 — standard
- **Adam**: adaptive learning rates per parameter, momentum — default choice
- **AdamW**: Adam with weight decay decoupled from gradient update — used for transformers

---

## Convolutional Neural Networks (CNNs)

Best for data with spatial/local structure: images, audio, some text tasks.

### Key Components
- **Convolutional layer**: apply a filter (kernel) that slides across the input, detecting local patterns
- **Pooling layer**: downsample the feature map (Max pooling, Average pooling)
- **Stride**: how many steps the filter moves at a time
- **Padding**: add zeros around input to control output size

### Famous CNN Architectures
| Architecture | Year | Key Contribution |
|---|---|---|
| LeNet | 1998 | First practical CNN for digit recognition |
| AlexNet | 2012 | Won ImageNet by large margin, triggered deep learning era |
| VGG | 2014 | Very deep (16–19 layers), simple blocks |
| ResNet | 2015 | Residual connections (skip connections) — enabled 100+ layers |
| EfficientNet | 2019 | Scales width/depth/resolution simultaneously |
| ConvNeXt | 2022 | CNN redesigned to match ViT performance |

### Residual Connections (ResNet)
```
output = F(x) + x
```
Adding the input directly to the output of a block lets gradients flow easily through very deep networks. This simple idea enabled training 100+ layer networks.

---

## Recurrent Neural Networks (RNNs)

Designed for sequential data (time series, text). Maintain a hidden state that is updated at each time step.

```
hₜ = tanh(W_h × hₜ₋₁ + W_x × xₜ + b)
```

**Problems with vanilla RNNs:**
- Vanishing gradients: gradients shrink to zero over long sequences
- Cannot parallelize — sequential by nature

### LSTM (Long Short-Term Memory)
Adds a cell state and three gates (forget, input, output) to regulate information flow:
- **Forget gate**: which information to discard from cell state
- **Input gate**: which new information to write to cell state
- **Output gate**: what to output from the cell state

LSTMs can remember information over hundreds of time steps.

### GRU (Gated Recurrent Unit)
Simplified LSTM with fewer parameters:
- Reset gate and update gate (no separate cell state)
- Slightly faster to train, often similar performance to LSTM

**Note**: RNNs have been largely superseded by Transformers for NLP, but remain relevant for streaming data and certain time series tasks.

---

## Normalization Techniques

### Batch Normalization
Normalize activations across the batch dimension:
```
x̂ = (x - μ_batch) / σ_batch
output = γ × x̂ + β
```
- γ, β are learned scale and shift parameters
- Accelerates training, allows higher learning rates
- Problems with small batch sizes, not suitable for RNNs or Transformers

### Layer Normalization
Normalize across the feature dimension instead of batch dimension:
- Works with any batch size, including batch size 1
- **Used in Transformers** (BERT, GPT, LLaMA)
- Applied before (pre-norm) or after (post-norm) attention layers

### RMSNorm
Simplified layer norm without mean centering:
- Faster to compute, similar performance
- Used in LLaMA, Mistral

---

## Regularization Techniques

### Dropout
Randomly set a fraction of neurons to 0 during training:
- Prevents co-adaptation of neurons
- Acts as an ensemble of many sub-networks
- Typical dropout rate: 0.1–0.5
- Should be disabled during inference

### L1 / L2 Regularization (Weight Decay)
Add penalty term to loss:
- **L1**: `L_total = L + λ Σ|w|` — encourages sparsity
- **L2**: `L_total = L + λ Σw²` — penalizes large weights, more common

### Early Stopping
Monitor validation loss; stop training when it starts increasing (overfitting).

### Data Augmentation
Artificially expand the training set:
- **Images**: random crop, flip, rotate, color jitter, mixup
- **Text**: back-translation, synonym replacement, random deletion/swap (EDA)
- **Audio**: time stretching, pitch shifting, noise addition

---

## Optimization Deep Dive

### Learning Rate
The most important hyperparameter. Too high → divergence; too low → slow convergence.

**Learning Rate Scheduling:**
- **Warm-up**: start with very low LR, gradually increase for first N steps (used in transformers)
- **Cosine annealing**: gradually decrease LR following a cosine curve
- **Reduce on plateau**: decrease LR when validation metric stops improving

### Gradient Clipping
Clip gradients to a maximum norm before the update:
```
if ||∇|| > threshold: ∇ = ∇ × threshold / ||∇||
```
Prevents exploding gradients — critical for RNNs and transformers.

### Mixed Precision Training
Use FP16 or BF16 for forward/backward passes, FP32 for weight updates:
- 2× faster on modern GPUs (tensor cores)
- Half the memory usage
- BF16 preferred over FP16 for LLMs (wider dynamic range)

---

## Generative Neural Networks

### Autoencoders
**Encoder**: compress input to a low-dimensional latent code z
**Decoder**: reconstruct the input from z

Applications: anomaly detection, denoising, representation learning

### Variational Autoencoders (VAEs)
The encoder outputs a distribution (μ, σ) instead of a single vector.
The decoder samples from this distribution → generative model.

### Generative Adversarial Networks (GANs)
Two networks in competition:
- **Generator G**: generate realistic-looking data from noise
- **Discriminator D**: distinguish real from generated data

Training: D tries to classify correctly; G tries to fool D.
Result: G learns to generate realistic images, audio, video.

Famous GAN variants: StyleGAN (photorealistic faces), CycleGAN (image-to-image translation), DCGAN

### Diffusion Models
Current SoTA for image generation:
- **Forward process**: gradually add Gaussian noise to data over T steps
- **Reverse process**: train a neural network to predict and remove noise at each step
- At inference: start from pure noise, repeatedly denoise

Used by: Stable Diffusion, DALL-E 3, Midjourney, Imagen

---

## Neural Network Debugging

### Loss is NaN
- Learning rate too high → gradient explode
- Log of zero (numerical instability) → add epsilon
- Missing batch norm → activations explodin

### Loss Not Decreasing
- Learning rate too low
- Vanishing gradients in deep net → use residual connections
- Bug in data loading (all same label?)
- Activations always negative → dying ReLU

### Overfitting
- Add dropout
- Add L2 regularization
- Data augmentation
- Reduce model size
- Get more data

### Debugging Tips
- Always overfit a small batch first (1–8 examples). If model can't memorize 8 examples, there's a model bug.
- Check data shapes at each layer
- Visualize gradients — should not be all zero or all exploding
- Log loss, accuracy, and learning rate to TensorBoard or wandb

---

## Modern Architecture Trends (2024)

- **State Space Models (SSMs)**: Mamba, RWKV — linear complexity alternatives to quadratic attention
- **Vision Transformers (ViT)**: patch-based attention applied to images — outperforms CNNs at scale
- **Graph Neural Networks (GNNs)**: GAT, GCN — for relational/graph-structured data
- **Neural Architecture Search (NAS)**: automated design of network architectures
- **Mixture of Experts (MoE)**: sparse activation, each input routes to different expert networks
- **Flash Attention 2**: memory-efficient exact attention enabling million-token context
