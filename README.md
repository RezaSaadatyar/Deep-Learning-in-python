
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
 - `Step 5. Fully Connected Layers:` After several convolutional and pooling layers, the feature maps are flattened into a one-dimensional vector. These flattened features are then passed to one or more fully connected layers, where each neuron is connected to every neuron in the previous and next layers. Fully connected layers are responsible for performing high-level feature representation and can be used for classification or regression tasks.
 - `Step 6. Output Layer:` The final fully connected
layer is typically followed by an output layer that produces the desired output. The number of neurons in the output layer depends on the specific task. For example, in image classification, the number of neurons in the output layer corresponds to the number of classes, and the output represents the probability distribution over the classes.
 - `Step 7. Loss Function and Optimization:` During
training, the CNN's output is compared to the true labels using a loss function, such as crossentropy. The loss function measures the discrepancy between the predicted output and the true output. The goal of the training process is to minimize this loss. Optimization algorithms, such as gradient descent, are used to iteratively adjust the weights of the network based on the computed gradients.
 - `Step 8. Backpropagation:` Backpropagation is used to compute and propagate gradients through the network. The gradients are used to update the weights in a direction that reduces the
loss. The process of forward propagation and backpropagation is repeated for a certain number of epochs to train the CNN.
 - `Step 9. Inference:` Once the CNN is trained, it can be used for inference on new, unseen data. The input data is passed through the network, and the output is obtained from the output layer.

 **Advantages of CNN:**<br/>
  - `Hierarchical Feature Learning:` CNNs are designed
to automatically learn hierarchical representations of features. Convolutional layers capture low-level features like edges and textures, while deeper layers capture higher-level features like shapes and object parts. This hierarchical feature learning
enables CNNs to extract meaningful representations from raw visual data.
 - `Spatial Invariance:` CNNs are able to achieve spatial invariance, meaning they can recognize patterns or features regardless of their location in the input data. This is achieved through the use of shared weights in convolutional layers, where the same filters are applied across the entire input, allowing the network to learn local patterns irrespective of their position.
 - `Parameter Sharing:` CNNs exploit the idea of parameter sharing, meaning that the same filters are applied to different spatial locations of the input. This significantly reduces the number of parameters in the network compared to fully connected networks, making CNNs more efficient in terms of memory usage and computational requirements.
 - `Translation Invariance:` CNNs are capable of achieving translation invariance, meaning they can recognize objects or features even if they are slightly shifted or translated in the input image. This is because the convolutional operation is performed at different locations across the input, allowing the network to capture local patterns irrespective of their precise location.
  - `Sparse Connectivity:` CNNs have sparse connectivity,
which means that each neuron is only connected to a small local region of the input. This sparse connectivity reduces the computational complexity of the network and enables efficient processing of large-scale visual data.
 - `Effective Parameter Learning:` CNNs leverage gradientbased
optimization techniques, such as backpropagation, to learn the optimal weights and parameters during training. This allows the network to adapt and improve its performance on specific tasks by minimizing the defined loss function.
 - `Transfer Learning:` CNNs trained on large datasets, such as ImageNet, can be used as feature extractors for new, smaller datasets. This transfer learning approach allows CNNs to leverage pre-trained knowledge and generalize well to new tasks even with limited training data.

 **Disadvantages of CNN**<br/>
  - `Computational Complexity:` CNNs can be computationally expensive to train and evaluate, especially for deeper networks with a large number of parameters. The high computational requirements may limit the deployment of CNNs on resource-constrained devices or in real-time applications.
 - `Need for Large Amounts of Data`: CNNs typically require large amounts of labeled training data to generalize well and achieve high performance. Acquiring and annotating such datasets can be time-consuming and costly, particularly in domains with limited data availability or in specialized tasks.
- `Lack of Interpretability:` CNNs are often considered black-box models, meaning that it can be challenging to interpret and understand the
decision-making process within the network. The complex hierarchical representations and a large number of parameters make it difficult to explain why a particular prediction or classification was made.
- `Vulnerability to Adversarial Attacks:` CNNs have been shown to be vulnerable to adversarial attacks, where small, imperceptible perturbations to input images can lead to incorrect or misleading predictions. Adversarial attacks pose security concerns, particularly in applications where robustness and reliability are crucial.
----
**A Recurrent Neural Network (RNN):**<br/> 
RNN is a type of neural network that is created to handle sequences of data. Unlike feedforward networks RNNs have connections that loop back, on themselves enabling them to retain a hidden state or memory of past inputs. This unique structure makes them highly suitable for tasks that involve data processing, such, as analyzing time series understanding language recognizing speech patterns and so on.

**RNNs possess features and elements:**
 - `Hidden State:` Within an RNN, the hidden state represents an enduring internal memory that extends across time steps, enabling the network to retain and recall information from preceding elements within the sequence.
 - `Recurrent Unit:` The fundamental component of an RNN, it accepts an input along with the previous hidden state, generating both an output and a new hidden state.
 - `Long Short-Term Memory (LSTM):` A variant of RNNs that introduces gating mechanisms to control the flow of information through the hidden state, enabling the network to capture long-term dependencies in sequential data.
 - `Gated Recurrent Unit (GRU):` Another variant of RNNs that also uses gating mechanisms but has a simpler architecture compared to LSTM.
- `Vanishing Gradient Problem:` A challenge in training RNNs where the gradients used for updating the network weights can become extremely small, leading to slow or ineffective learning.
- `Exploding Gradient Problem:` A challenge in training RNNs where the gradients become extremely large, causing instability during the learning process.






