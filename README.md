# Basic CNN From Scratch (NumPy)

A simple Convolutional Neural Network (CNN) built completely from scratch using only
Python and NumPy.

This project trains a small CNN on the MNIST handwritten digit dataset without using
high-level deep learning frameworks like TensorFlow or PyTorch for the actual network
implementation.

The goal of this repository is educational:

```
● Understand how CNNs actually work internally
● Learn the mathematics behind convolutions and backpropagation
● Build intuition for feature extraction and image classification
● Implement core deep learning concepts manually
```
# What This Project Does

This model:

```
● Loads the MNIST handwritten digit dataset
● Normalizes image data
● One-hot encodes labels
● Performs convolution operations manually
● Uses pooling layers for dimensionality reduction
● Applies activation functions
● Runs forward propagation
● Computes loss and gradients
● Updates weights using gradient descent
● Predicts handwritten digits
```
The implementation is intentionally low-level to show how CNNs work under the hood.


# Introduction to CNNs

A Convolutional Neural Network (CNN) is a type of neural network designed primarily
for image data.

Traditional neural networks flatten images into long vectors, which causes them to lose
spatial information.
CNNs solve this problem by learning spatial patterns directly from the image.

CNNs are extremely effective for:

```
● Image classification
● Object detection
● Facial recognition
● Medical imaging
● Self-driving car vision systems
● OCR (optical character recognition)
```
The MNIST dataset is commonly used as a beginner dataset because it contains
grayscale images of handwritten digits from 0–9.

# How CNNs Work

A CNN typically contains several major components:

1. Convolution Layers
2. Activation Functions
3. Pooling Layers
4. Fully Connected Layers
5. Output Layer

# 1. Convolution Layer

The convolution layer is the core of a CNN.


A small matrix called a kernel (or filter) slides across the image and extracts patterns
such as:

```
● edges
● curves
● textures
● shapes
```
Mathematically, convolution is:

$$
(I \ast K)(i, j) = \sum_m \sum_n I(i - m, j - n) K(m, n)
$$

Where:

```
● (I) is the input image
● (K) is the kernel/filter
● ((i,j)) is the output pixel location
```
The kernel multiplies overlapping image values and sums them together.

This creates a feature map.

# Example of a Convolution Kernel

A simple edge detection kernel:

$$
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

This filter highlights horizontal edges in an image.

Different kernels learn different visual features during training.

# 2. Activation Functions

After convolution, the network applies a non-linear activation function.


One common activation function is ReLU:

$$
f(x) = \max(0, x)
$$

ReLU removes negative values and introduces non-linearity.

Without activation functions, the network could only learn linear relationships.

# 3. Pooling Layer

Pooling reduces the spatial size of feature maps.

This helps:

```
● reduce computation
● reduce overfitting
● keep important features
```
Max pooling takes the largest value from a small region.

Example:

Input:

[
\begin{bmatrix}
1 & 2 \
5 & 3
\end{bmatrix}
]

Output after max pooling:

[
5
]


# 4. Fully Connected Layer

After feature extraction, the network flattens the feature maps into a vector.

A dense layer then performs classification.

The output is computed as:

$$
y = \mathbf{W}x + b
$$
Where:

```
● (W) = weights
● (x) = input vector
● (b) = bias
```
# 5. Softmax Output

The final layer converts outputs into probabilities.

Softmax equation:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The class with the highest probability becomes the prediction.

# Training Process

The CNN learns through gradient descent.

## Step 1 — Forward Pass

The image moves through the network.


## Step 2 — Loss Calculation

The prediction is compared to the correct label.

A common loss function is cross entropy:

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

## Step 3 — Backpropagation

Gradients are calculated.

## Step 4 — Weight Updates

Weights are updated using:

$$
w := w - \eta \frac{\partial L}{\partial w}
$$

Where:

```
● (\eta) is the learning rate
● (L) is the loss function
```
The network slowly improves over many iterations.

# Architecture Used In This Project

This implementation includes:

```
● Two convolution layers
● Multiple learnable kernels
● Pooling layers
● Manual forward propagation
● Manual training loop
● NumPy-only math operations
● Gradient descent optimization
```

The kernels are initialized using Kaiming initialization:

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n}}\right)
$$

This helps stabilize training.

# Dataset

This project uses the MNIST dataset.

MNIST contains:

```
● 60,000 training images
● 10,000 testing images
● 28x28 grayscale handwritten digits
```
The dataset is loaded using:

from keras.datasets import mnist

# Example Usage

x_small = x_train[:500]
y_small = y_train_oh[:500]

model = CNN(
x_small[0],
kernal_size=3,
layer_one_size=4,
layer_two_size=8,
learning_rate=0.
)

model.fit(x_small, y_small, epochs=10)


Predicting a digit:

probs = model.predict(x_test[0])
predicted_class = np.argmax(probs)

# Why Build A CNN From Scratch?

Frameworks like PyTorch and TensorFlow automate most of the process.

Building one manually teaches:

```
● tensor operations
● convolutions
● gradient flow
● optimization
● numerical stability
● matrix calculus
● feature extraction
```
This creates a much deeper understanding of deep learning.


# Technologies Used

```
● Python
● NumPy
● Keras (dataset only)
```
# Educational Purpose

This repository is designed primarily for learning and experimentation.

The implementation prioritizes clarity and understanding over speed.


