# Basic CNN From Scratch (NumPy)

A simple Convolutional Neural Network (CNN) built completely from scratch using only
Python and NumPy.

This project trains a small CNN on the MNIST handwritten digit dataset without using
high-level deep learning frameworks like TensorFlow or PyTorch for the actual network
implementation.

The goal of this repository is educational:

* Understand how CNNs actually work internally
* Learn the mathematics behind convolutions and backpropagation
* Build intuition for feature extraction and image classification
* Implement core deep learning concepts manually

# What This Project Does

This model:


* Loads the MNIST handwritten digit dataset
* Normalizes image data
* One-hot encodes labels
* Performs convolution operations manually
* Uses pooling layers for dimensionality reduction
* Applies activation functions
* Runs forward propagation
* Computes loss and gradients
* Updates weights using gradient descent
* Predicts handwritten digits

The implementation is intentionally low-level to show how CNNs work under the hood.


# Introduction to CNNs

A Convolutional Neural Network (CNN) is a type of neural network designed primarily
for image data.

Traditional neural networks flatten images into long vectors, which causes them to lose
spatial information.
CNNs solve this problem by learning spatial patterns directly from the image.

CNNs are extremely effective for:


* Image classification
* Object detection
* Facial recognition
* Medical imaging
* Self-driving car vision systems
* OCR (optical character recognition)

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


* edges
* curves
* textures
* shapes

Mathematically, convolution is:

$$
(I \ast K)(i, j) = \sum_m \sum_n I(i - m, j - n) K(m, n)
$$

Where:


* (I) is the input image
* (K) is the kernel/filter
* ((i,j)) is the output pixel location

The kernel multiplies overlapping image values and sums them together.

This creates a feature map.

Basic convolution function example:
```python
import numpy as np

def convolution(input_array, kernel):
  output = []

  # Get kernel dimensions (height × width)
  kernel_h, kernel_w = kernel.shape

  # Slide the kernel over the input vertically
  for i in range(input_array.shape[0] - kernel_h + 1):
    row = []

    # Slide the kernel over the input horizontally
    for j in range(input_array.shape[1] - kernel_w + 1):

      # Extract a local region (same size as kernel)
      # This is a matrix slice of the input
      block = input_array[i:i+kernel_h, j:j+kernel_w]

      # Element-wise multiplication:
      # (input patch) ⊙ (kernel)
      #
      # Then sum all results:
      # Σ (block[i,j] * kernel[i,j])
      #
      # This is the core convolution operation
      val = np.sum(block * kernel)

      # Store single scalar output for this position
      row.append(val)

    # One full row of convolved outputs
    output.append(row)

  # Convert list of lists into 2D array (feature map)
  return np.array(output)
```

# Example of a Convolution Kernel

A simple edge detection kernel:

$$
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1
\end{bmatrix}
$$

Coding example:
```python
import numpy as np

arr = np.array([
      [1, 2, 3, 0],
      [4, 5, 6, 1],
      [7, 8, 9, 2],
      [0, 1, 2, 3]
    ])

kernel = np.array([
      [1, 0],
      [0, -1]
    ])

print(convolution(arr, kernel))
```
Output:
```python
[[-4 -4  2]
 [-4 -4  4]
 [ 6  6  6]]
```
# 2. Activation Functions

After convolution, the network applies a non-linear activation function.


One common activation function is ReLU:

$$
f(x) = \max(0, x)
$$

ReLU removes negative values and introduces non-linearity.

Without activation functions, the network could only learn linear relationships.

Coding example: 
```python
import numpy as np

def RELU(input_array):
  #if element in input_array is less than zero replace it with zero
  return np.maximum(0, input_array) 
```
# 3. Pooling Layer

Pooling reduces the spatial size of feature maps.

This helps:


* reduce computation
* reduce overfitting
* keep important features

Max pooling takes the largest value from a small region.

Example:

Input:

$$
\begin{bmatrix}
1 & 2 \\
5 & 3
\end{bmatrix}
$$

Output after max pooling:

$$
[
5
]
$$

Coding example of pooling function:
```python
import numpy as np

def max_pooling(input_array, pooling_size):
  # Get input dimensions
  rows, cols = input_array.shape

  # Compute output dimensions after pooling
  # (each pooling window collapses into 1 value)
  out_rows = rows // pooling_size
  out_cols = cols // pooling_size

  # Output feature map (stores max values)
  output = np.zeros((out_rows, out_cols))

  # Mask to track where max values came from (used in backprop in CNNs)
  mask = np.zeros_like(input_array)

  # Slide pooling window vertically
  for i in range(out_rows):
    # Slide pooling window horizontally
    for j in range(out_cols):

      # Extract pooling region (pooling_size × pooling_size block)
      block = input_array[
          i * pooling_size:(i + 1) * pooling_size,
          j * pooling_size:(j + 1) * pooling_size
      ]

      # MAX operation:
      # mathematically: max(x_ij in block)
      # keeps only the strongest activation in this region
      max_val = np.max(block)

      # Store pooled result in output feature map
      output[i, j] = max_val

      # Create binary mask showing where max occurred
      # (1 at max location, 0 elsewhere)
      block_mask = (block == max_val)

      # Place mask back into correct location in full-sized mask
      mask[
          i * pooling_size:(i + 1) * pooling_size,
          j * pooling_size:(j + 1) * pooling_size
      ] = block_mask

  # Return:
  # - pooled feature map (downsampled)
  # - mask (used for gradient flow in CNN backprop)
  return output, mask
```
# 4. Fully Connected Layer

After feature extraction, the network flattens the feature maps into a vector.

A dense layer then performs classification.

The output is computed as:

$$
y = \mathbf{W}x + b
$$
Where:


* (W) = weights
* (x) = input vector
* (b) = bias

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


* (\eta) is the learning rate
* (L) is the loss function

The network slowly improves over many iterations.

# Architecture Used In This Project

This implementation includes:


* Two convolution layers
* Multiple learnable kernels
* Pooling layers
* Manual forward propagation
* Manual training loop
* NumPy-only math operations
* Gradient descent optimization


The kernels are initialized using Kaiming initialization:

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n}}\right)
$$

This helps stabilize training.

# Dataset

This project uses the famouse MNIST dataset. 

MNIST contains:


* 60,000 training images
* 10,000 testing images
* 28x28 grayscale handwritten digits

# Why Build A CNN From Scratch?

Frameworks like PyTorch and TensorFlow automate most of the process.

Building one manually teaches:


* tensor operations
* convolutions
* gradient flow
* optimization
* numerical stability
* matrix calculus
* feature extraction

This gives a more solid base around CNNs.


# Technologies Used


* Python
* NumPy
* Keras (dataset only)

# Educational Purpose

This repository is designed primarily for learning and experimentation.

The implementation prioritizes clarity and understanding over speed.

Full CNN class is on file basicCNN.ipynb. 

