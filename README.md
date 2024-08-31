**Deep Learning in Python**

[View Jupyter Notebook][def]


**Creating Deep Learning model step by step**<br/>
- `Data Preprocessing:`
  - *Cleaning:* Handle missing values, remove duplicates, and correct errors.
  - *Normalization/Standardization:* Scale numerical data to improve the model's performance.
  - *Encoding:* Convert categorical data into numerical format (e.g., `one-hot encoding` or `label encoding`).
  - *Augmentation (for images):* Apply transformations like rotations, flips, and shifts to artificially expand the dataset.
- `Data Splitting:`<br/>
   Divide the data into training, validation, and test sets (e.g., 70% training, 15% validation, 15% testing).
- `Choose a Model Architecture (Select Model Type):`
  - *Feedforward Neural Networks (FNNs)* for tabular data.
  - *Convolutional Neural Networks (CNNs)* for image classification.
  - *Recurrent Neural Networks (RNNs)* or *Long Short-Term Memory (LSTM)* networks for time series or sequence data.
  - *Transformers* for language models or advanced sequence processing.
- `Define Layers:`<br/> 
   Determine the number of layers, *types of layers* (`Dense, Conv2D`, `LSTM`, etc.), *activation functions* (`ReLU`, `Sigmoid`, `Softmax`), and *layer configurations*.
- `Configure the Model (Compile the Model):`
  - *Loss Function:* Choose an appropriate loss function (e.g., `categorical cross-entropy for classification`, `mean squared error` for regression).
  - *Optimizer:* Select an optimizer (e.g., `SGD`,` Adam`) to update model weights during training.
  - *Metrics:* Specify metrics to evaluate the model's performance (e.g., `accuracy`, `precision`).
- `Train the Model:`
   - *Set Hyperparameters:*
      - Batch Size: Number of samples processed before the model is updated.
      - Epochs: Number of complete passes through the training dataset.
      - Learning Rate: Controls how much to adjust the weights during training.
   - *Training Process:*
      - Fit the model on the training data using the `.fit()` method (if using libraries like TensorFlow/Keras).
      - Use validation data to tune the model and check for overfitting or underfitting.
   - *Monitor Training:*
      Use callbacks like *Early Stopping*, *Model Checkpoint*, or *Learning Rate Schedulers* to improve training efficiency.
- `Evaluate the Mode:`
   - *Performance on Test Data:* Evaluate the model on the test set to check how well it generalizes to unseen data.
   - *Metrics Analysis:* Review performance metrics such as accuracy, loss, confusion matrix, ROC-AUC for classification, or R-squared for regression.
   - *Error Analysis:* Identify specific instances where the model fails to perform well and understand the reasons.
- `Tune the Model (Hyperparameter Tuning):`
   - *Adjust Hyperparameters:* Experiment with different hyperparameters (`learning rate`, `batch size`, `number of layers`, `neurons per layer`, etc.).
   - *Cross-Validation:* Use techniques like `k-fold cross-validation` to better assess model performance.
   - *Automated Tuning:* Utilize tools like `Grid Search`, `Random Search`, or `Bayesian Optimization` for systematic hyperparameter tuning.
- `Deploy the Model:`
   - *Model Export:* Save the trained model in a format suitable for deployment (e.g., `HDF5`, `SavedModel`).
   - *Deployment Options:* Choose how to deploy the model (e.g., as a `REST API using Flask/Django`, `using cloud services like AWS, Azure, or GCP`).


- [x] **Backpropagation (Backward propagation of errors):**<br/>
Backpropagation is an algorithm for supervised learning of artificial neural networks using gradient descent. This is because artificial neural networks use backpropagation as a learning algorithm to compute a gradient descent with respect to weight values for the various inputs. By comparing desired outputs to achieved system outputs, the systems are tuned by adjusting connection weights to narrow the difference between the two as much as possible. It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.
![image](https://github.com/RezaSaadatyar/Deep-Learning-in-python/assets/96347878/b08a633b-4120-4603-b4a1-bf07301de1ad)

Having too much information<br/>
A sufficient number of computing resources<br/>

Complexity issues:
 - Object Detection
 - Speech Recognition
 - Image Segmentation
 - Image Classification
 - Reinforcement Learning
 - Natural Language Processing

**Wide and Deep Network** 


![image](https://user-images.githubusercontent.com/96347878/202599415-21af41e0-3d0d-46b0-9f9a-1384879fe6c0.png) ![image](https://user-images.githubusercontent.com/96347878/202598550-cca36b19-da51-4849-a590-4f848ae4e898.png) ![image](https://user-images.githubusercontent.com/96347878/202598209-f2a7aceb-a6c1-4698-97fd-3c705d19e5dd.png)

----
**Convolutional Neural Network (CNN):**<br/>
A CNN is a type of deep learning algorithm that is widely used for processing and analyzing visual data such as images and videos. CNNs consist of multiple
layers of interconnected neurons, including convolutional layers, pooling layers, and fully connected layers.<br/>
 - `Convolution:` Applying a filter or kernel to input data is a mathematical process that aids in feature extraction. This is accomplished by multiplying each element and then summing them.
 - `Convolutional Layer:` A CNN layer applies convolutional operations to the input data by using one or multiple filters/kernels.
- `Filter/Kernel:` A small matrix of weights that is
convolved with the input data to produce a
feature map.
 - `Feature Map:` The output of a convolutional layer, which represents the presence of specific
features in the input data.
 - `Pooling: `This refers to a downsampling process that decreases the feature maps' spatial dimensions by picking out the most crucial or representative values.
 - `Pooling Layer:` In a CNN, there is a layer that conducts pooling operations to decrease the spatial dimensions and capture significant characteristics from the feature maps.
 - `Stride:` The step size used to move the filter/kernel during the convolution operation.
 - `Padding:` Adding additional pixels or values around the input data to preserve spatial information during convolution.

**What Is the Mechanism Behind CNNs?**
 - `Step 1, Input Layer:` In a CNN, input usually consists of one or more images. These images are represented as a matrix of pixels, with each pixel holding either color data (in RGB format) or grayscale intensity.
 - `Step 2. Convolutional Layers:` CNN starts by utilizing one or more convolutional layers. Each layer is composed of numerous filters or kernels, which are matrices of weights. These filters are applied to the input data using a sliding window technique. By performing element-wise multiplication and summation, the filters are able to identify particular features or patterns.
  - `Step 3. Activation Function:` Following the convolution operation, a non-linear activation function, like ReLU (Rectified Linear Unit), is applied to each element in order to introduce non-linearity.
 - `Step 4. Pooling Layers:` Pooling layers are incorporated into the network to decrease the spatial dimensions of the feature maps and extract crucial information. Max pooling or average pooling are commonly used pooling operations. The purpose of pooling is to reduce the number of parameters and computational complexity, while still preserving the most significant features.
 - `Step 5. Fully Connected Layers:` After convolutional and pooling layers, feature maps are flattened into a 1D vector. These flattened features are then passed to fully connected layers, where each neuron is connected to every neuron in the previous and next layers. Fully connected layers perform high-level feature representation and can be used for classification or regression tasks.
 - `Step 6. Output Layer:` The last fully connected layer is usually followed by an output layer that generates the desired output. The number of neurons in the output layer varies depending on the task. In image classification, for instance, the number of neurons corresponds to the number of classes, and the output shows the probability distribution across the classes.
 - `Step 7. Loss Function and Optimization:` In the process of training a CNN, a loss function like crossentropy is utilized to compare its output with the actual labels. This function determines the difference between the predicted and true output. The primary objective of the training is to reduce this loss. To achieve this, optimization algorithms such as gradient descent are employed to gradually modify the network's weights based on calculated gradients.
 - `Step 8. Backpropagation:` The purpose of backpropagation is to calculate and transmit gradients throughout the network. These gradients are then utilized to adjust the weights in a way that minimizes the loss. This process of forward propagation and backpropagation is iterated for a specific number of epochs in order to train the CNN.
 - `Step 9. Inference:` After the training of CNN, it becomes applicable for making inferences on new and unseen data. The network processes the input data, and the output is generated from the output layer.

 **Advantages of CNN:**<br/>
  - `Hierarchical Feature Learning:` The purpose of CNNs is to learn hierarchical representations of features in an automated manner. The lower-level features such as edges and textures are captured by the convolutional layers, while the deeper layers capture higher-level features such as shapes and object parts. This hierarchical feature learning process enables CNNs to extract significant representations from raw visual data.
 - `Spatial Invariance:` The utilization of shared weights in convolutional layers enables CNNs to attain spatial invariance, which implies that they can detect features or patterns regardless of their position in the input data. This technique allows the network to learn local patterns without considering their location.
 - `Parameter Sharing:` The concept of parameter sharing is utilized by CNNs, where identical filters are used for various spatial positions of the input. This results in a considerable decrease in the number of parameters present in the network as compared to fully connected networks, making CNNs more memory-efficient and computationally less demanding.
 - `Translation Invariance:` Translation invariance can be achieved by CNNs, which enables them to identify objects or features even if they are slightly displaced or moved in the input image. The convolutional operation is carried out at various positions throughout the input, allowing the network to detect local patterns regardless of their exact location.
  - `Sparse Connectivity:` The connectivity in CNNs is sparse, implying that each neuron solely links to a small local area of the input. This sparsity in connectivity reduces the network's computational complexity, allowing for efficient handling of visual data on a large scale.
 - `Effective Parameter Learning:` CNNs make use of gradient-based optimization methods, such as backpropagation, to iteratively adjust and optimize the weights and parameters of the network during the training process. This enables the network to learn and enhance its performance on particular tasks by reducing the defined loss function.
 - `Transfer Learning:` CNNs trained on large datasets like ImageNet can be used as feature extractors for smaller datasets. This transfer learning approach allows CNNs to generalize well to new tasks even with limited training data by leveraging pre-trained knowledge.

 **Disadvantages of CNN**<br/>
  - `Computational Complexity:` CNNs are costly to train and evaluate, particularly for deeper networks with many parameters. This may restrict their use in real-time applications or on devices with limited resources.
 - `Need for Large Amounts of Data`: CNNs need lots of labeled data to perform well. Getting and labeling such data can be expensive and time-consuming, especially in domains with limited data or specialized tasks.
- `Lack of Interpretability:` CNNs are seen as black-box models, making it hard to interpret the decision-making process within the network due to the complex hierarchical representations and numerous parameters, which makes it difficult to explain why a specific prediction or classification was made.
- `Vulnerability to Adversarial Attacks:` CNNs are vulnerable to adversarial attacks, where slight changes to input images can cause incorrect predictions. This is a security concern in applications where reliability is important.
----
**A Recurrent Neural Network (RNN):**<br/> 
RNN is a type of neural network that is created to handle sequences of data. Unlike feedforward networks RNNs have connections that loop back, on themselves enabling them to retain a hidden state or memory of past inputs. This unique structure makes them highly suitable for tasks that involve data processing, such, as analyzing time series understanding language recognizing speech patterns and so on.

**RNNs possess features and elements:**
 - `Hidden State:` Within an RNN, the hidden state represents an enduring internal memory that extends across time steps, enabling the network to retain and recall information from preceding elements within the sequence.
 - `Recurrent Unit:` The fundamental component of an RNN, it accepts an input along with the previous hidden state, generating both an output and a new hidden state.
- `Vanishing Gradient Problem:` It refers to the difficulty encountered when training RNNs due to the gradients used for updating the network weights becoming exceedingly small. This can result in slow or ineffective learning..
- `Exploding Gradient Problem:` The issue of Exploding Gradient arises when training RNNs as the gradients become too large, leading to instability in the learning process.
- `Sequence-to-Sequence (Seq2Seq):` The Seq2Seq framework is based on RNNs and is utilized for tasks that require sequential input and output, like text generation or machine translation.
 - `Bidirectional RNN:` It is a type of RNN that operates on the input sequence in both forward and backward directions. This enables the network to comprehend connections from both previous and forthcoming elements.
- `Time Step:` A discrete point in time in the sequential data being processed by an RNN.
- `Teacher Forcing:` A frequently employed training technique for RNNs, wherein actual output values are reintroduced as inputs, as opposed to predicted values. This approach facilitates the acquisition of long-range dependencies during training

**Types of RNNs:** 
- `Long Short-Term Memory (LSTM):` LSTM is a type of RNN that uses gating mechanisms to regulate the flow of information in the hidden state. This allows the network to effectively capture long-term patterns in sequential data.
 - `Gated Recurrent Unit (GRU):` GRU is another type of RNN that incorporates gating mechanisms, but it has a more straightforward structure when compared to LSTM.

**What is the mechanism behind the functioning of RNN?**
- `Step 1. Input Preparation:` The first step involves preparing the input by encoding each element in the sequential data as a feature vector or embedding, such as words in a sentence or time steps in a time series. These input vectors are then sequentially fed into the RNN one at a time.
- `Step 2. Hidden State Initialization:` In the second step of processing for an RNN, the hidden state is initialized before the first element is processed. This initialization is usually done by setting the hidden state to either zero or small random values. The purpose of the hidden state is to act as the memory for the network and store information from previous elements in the sequence.
- `Step 3. Recurrent Unit Operation:` During each time step, the input vector and the previous hidden state are merged and then passed into the recurrent unit. The recurrent unit then utilizes the input and hidden state to carry out computations, resulting in an output and a modified hidden state.
- `Step 4. Output Generation:` Depending on the task at hand, the recurrent unit's output at the present time step can serve different purposes. For instance, in language modeling, it can be utilized for forecasting the following word in the sequence.
- `Step 5. Hidden State Update:` The hidden state of the recurrent unit is updated and carried over to the next time step, enabling the RNN to preserve past information and comprehend temporal dependencies.
- `Step 6. Repeat for Remaining Elements:`The process of steps 3-5 is iterated for every element present in the sequential data, enabling the RNN to handle the complete sequence.
- `Step 7. Output Utilization:` Once all the elements are processed, the RNN can utilize either the final hidden state or the sequence of outputs to perform the intended task, be it prediction, classification, or generation of new sequences.

**Advantages of RNN:**
 - `Capturing Sequential Dependencies:`RNNs are great at capturing patterns in sequential data by maintaining an internal memory. They can remember and use information from previous elements in the sequence, making them effective for tasks that require understanding the order and context of the data.
 - `Variable Length Input/Output:`RNNs process variable-length input and output sequences, making them useful in natural language processing where sentences have varying word counts. Unlike fixed-size input networks, RNNs can handle different input lengths.
 - `Parameter Sharing:`RNNs reuse the knowledge learned from earlier elements in the sequence through parameter sharing, making them more efficient and reducing the number of parameters needed to process a sequence compared to models that treat each time step independently.
 - `Language Modeling and Sequence Generation:`RNNs are often used for language modeling, predicting the next word, or generating new text. They are effective at capturing dependencies and context, resulting in coherent and relevant sequences.

**Disadvantages of RNN:** 
 - `Vanishing and Exploding Gradients:`Training RNNs can be challenging due to vanishing or exploding gradients. Backpropagating gradients through many time steps can cause extremely small or large signals, making it difficult to learn long-term dependencies or causing instability in training.
 - `Computational Complexity`:RNNs can be computationally expensive for long sequences or large hidden state sizes. The sequential nature of RNNs makes it hard to parallelize computations across time steps, slowing down the training process and limiting scalability.
 - `Sensitivity to Input Order:`RNNs are sensitive to input sequence order and even a small change can lead to different outputs. This makes RNNs more prone to noise or variations in input data.
 - `Difficulty in Capturing Long-Term Dependencies:`RNNs can have difficulty capturing long-term dependencies, as the gradient signal may weaken and become ineffective in propagating information located far back in the sequence. This limitation can affect the model's ability to model long-range dependencies.
- `Lack of Attention Mechanism:`RNNs lack a mechanism to focus on specific parts of input sequences, making it difficult for them to handle long or important sequences effectively.
----


[def]: [URL_TO_YOUR_NOTEBOOK](https://github.com/RezaSaadatyar/Deep-Learning-in-python/blob/main/Code/Deep_Learning.ipynb)