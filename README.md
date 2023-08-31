
#### Deep Learning in python

- [x] **Backpropagation (Backward propagation of errors):**<br/>
Backpropagation is an algorithm for supervised learning of artificial neural networks using gradient descent.This is because artificial neural networks use backpropagation as a learning algorithm to compute a gradient descent with respect to weight values for the various inputs. By comparing desired outputs to achieved system outputs, the systems are tuned by adjusting connection weights to narrow the difference between the two as much as possible. It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.
![image](https://github.com/RezaSaadatyar/Deep-Learning-in-python/assets/96347878/b08a633b-4120-4603-b4a1-bf07301de1ad)



Having too much information<br/>
A sufficient number of computing resources<br/>

Complexity issues:
 - Object Detection
 - Speech Recognition
 - Image Segmentation
 - Image Classification
 - Reinforement Laerning
 - Natural Language Processing

**Wide and Deep Network** 


![image](https://user-images.githubusercontent.com/96347878/202599415-21af41e0-3d0d-46b0-9f9a-1384879fe6c0.png)

![image](https://user-images.githubusercontent.com/96347878/202598550-cca36b19-da51-4849-a590-4f848ae4e898.png)


![image](https://user-images.githubusercontent.com/96347878/202598209-f2a7aceb-a6c1-4698-97fd-3c705d19e5dd.png)
----
**Convolutional Neural Network (CNN):**<br/>
A CNN is a type of deep learning algorithm that is widely used for processing and analyzing visual data such as images and videos. CNNs consist of multiple
layers of interconnected neurons, including convolutional layers, pooling layers, and fully connected layers.<br/>
 - `Convolution:` The mathematical operation of
applying a filter/kernel to the input data, which
helps extract features by performing elementwise
multiplication and summation.
 - `Convolutional Layer:` The layer in a CNN that
applies convolutional operations to the input data
using one or more filters/kernels.
- `Filter/Kernel:` A small matrix of weights that is
convolved with the input data to produce a
feature map.
 - `Feature Map:` The output of a convolutional layer, which represents the presence of specific
features in the input data.
 - `Pooling: `A downsampling operation that reduces
the spatial dimensions of the feature maps by
selecting the most important or representative
values.
 - `Pooling Layer:` The layer in a CNN that performs
pooling operations to reduce the spatial
dimensions and extract salient features from the
feature maps.
 - `Stride:` The step size used to move the
filter/kernel during the convolution operation.
 - `Padding:` Adding additional pixels or values
around the input data to preserve spatial
information during convolution.<br/>

**What Is the Mechanism Behind CNNs?**
 - `Step 1, Input Layer:` The input to the CNN is typically an image or a set of images. Each image is
represented as a grid of pixels, where each pixel
contains color information (RGB values) or grayscale
intensity.
 - `Step 2. Convolutional Layers:` The CNN begins with one or more convolutional layers. Each convolutional layer consists of multiple filters/kernels, which are small matrices of weights. These filters are convolved with the input data using a sliding window approach. The filters detect specific features or patterns by performing element-wise multiplication and summation.
  - `Step 3. Activation Function:` After the convolution operation, an activation function, such as ReLU (Rectified Linear Unit), is applied element-wise to introduce non-linearity.
 - `Step 4. Pooling Layers:` To reduce the spatial
dimensions of the feature maps and extract the most important information, pooling layers are added. Common types of pooling operations include max pooling or average pooling. Pooling helps reduce the number of parameters and computational complexity while retaining the most salient features.

