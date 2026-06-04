# 🧠 Convolutional Neural Network — Built From Scratch

> No PyTorch. No TensorFlow. No shortcuts. Just NumPy and a deep understanding of what a neural network is actually doing.

---

## What Is This?

A fully functional Convolutional Neural Network implemented from scratch using only **NumPy**. Every single component — forward pass, backward pass, weight updates — was written by hand without any deep learning framework doing the heavy lifting.

Built as part of my **BSc Mathematics** degree at **Queen Mary University of London**, the goal was simple: understand what is happening at every layer before trusting a library to abstract it away.

---

## Why Build It From Scratch?

Most deep learning tutorials teach you to call `model.fit()` and move on. That tells you nothing about what is actually happening inside the network. Building from scratch forces you to understand:

- How convolutions actually slide across an image and extract features
- What backpropagation is doing mathematically at every layer
- Why activation functions exist and what happens without them
- How gradients flow backwards through pooling, convolutional and fully connected layers

If you can build it from scratch, you actually understand it.

---

## Architecture

```
Input → Conv Layer → ReLU → Max Pooling → Fully Connected → Softmax → Output
```

| Component | Details |
|---|---|
| Convolutional Layer | Custom kernel, stride and padding — implemented manually |
| Activation | ReLU — applied element-wise, gradient computed by hand |
| Pooling | Max Pooling — forward and backward pass both implemented |
| Fully Connected | Dense layer with manual weight initialisation |
| Output | Softmax classifier |
| Loss | Cross-entropy loss |
| Optimiser | Stochastic Gradient Descent (SGD) |

---

## Implementation Details

**Forward Pass**
- Convolution operation implemented as a sliding window across input feature maps
- ReLU activation zeroing negative values element-wise
- Max pooling selecting maximum values across pooling windows
- Fully connected layer computing weighted sum with bias
- Softmax converting raw scores to class probabilities

**Backward Pass**
- Cross-entropy gradient flowing back through softmax
- Fully connected layer gradients computed with chain rule
- Max pooling gradients routed only through max-value positions
- Convolutional layer gradients computed for both filters and inputs
- Weights updated via SGD

---

## Performance Evaluation

Model evaluated using the following metrics:

- **Accuracy** — overall correct classifications
- **Precision** — true positives out of all predicted positives
- **Recall** — true positives out of all actual positives
- **F1-Score** — harmonic mean of precision and recall
- **Validation curves** — to monitor overfitting across epochs

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| NumPy | All matrix operations and computations |
| torchvision | Dataset loading only |
| Matplotlib | Visualisation of training curves and results |

---

## How To Run

```bash
# Clone the repository
git clone https://github.com/yourusername/cnn-from-scratch

# Install dependencies
pip install numpy torchvision matplotlib

# Run the notebook
jupyter notebook CNN_From_Scratch.ipynb
```

---

## What I Learned

- The maths behind convolutional operations and how filters learn spatial features
- How backpropagation works layer by layer using the chain rule
- Why weight initialisation matters and how poor initialisation kills training
- The difference between training, validation and test performance
- How hyperparameter choices — learning rate, kernel size, pooling size — affect convergence

---
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

*Part of an ongoing portfolio building toward a career in machine learning and AI engineering.*
