# Deep-learning-basics
CIFAR-10 Classification with Various CNN networks

# CIFAR-10 Experiments
In this and the upcoming segments, you will experiment with a few `hyperparameters and architectures and draw insights from the results`. Some hyperparameters you will play with are listed below:

- Adding and removing dropouts in convolutional layers
- Batch normalisation
- L2 regularisation
- Increasing the number of convolution layers
- Increasing the number of filters in certain layers

# 1. Using Dropouts after the convolution and FC layers

In the first experiment, you will use ‘dropouts’ both after the convolutional and FC layers.


# 2. Batch Normalisation
​
**Why Batch Normalization**
​
One of the most common problems of data science professionals is to `avoid over-fitting`. Have you come across a situation when your model is performing very well on the training data but is unable to predict the test data accurately. `The reason is your model is overfitting`. The **solution to such a problem is regularization.
​
`The regularization techniques` ``help to improve a model and allows it to converge faster``. We have several regularization tools at our end, some of them are early stopping, dropout, weight initialization techniques, and batch normalization. `The regularization helps in preventing the over-fitting of the model and the learning process becomes more efficient`.
​
**What is Batch Normalization?**
​
Before entering into Batch normalization let’s understand the term “Normalization”.
​
Normalization is a data pre-processing tool used to bring the numerical data to a common scale without distorting its shape.
​
Generally, when we input the data to a machine or deep learning algorithm we tend to change the values to a  balanced scale. The reason we normalize is partly to ensure that our model can generalize appropriately.
​
Now coming back to Batch normalization, `it is a process to make neural networks faster and more stable through adding extra layers in a deep neural network. `The new layer performs the standardizing and normalizing operations on the input of a layer coming from a previous layer.
​
But what is the reason behind the term “Batch” in batch normalization? A typical neural network is trained using a collected set of input data called batch. Similarly, the normalizing process in batch normalization takes place in batches, not as a single input
​
**Here's how batch normalization works:**
​
- During training, for each mini-batch of inputs, batch normalization computes the mean and standard deviation of the activations within that batch.
​
- It then normalizes the activations by subtracting the batch mean and dividing by the batch standard deviation. This centers the activations around zero and scales them to have unit variance.
​
- After normalization, batch normalization applies a scale factor (gamma) and an offset (beta) to the normalized activations. These parameters are learned during training and allow the network to adapt the normalized values if needed.
​
- The normalized and scaled activations are then passed through a non-linear activation function (e.g., ReLU) and fed to the next layer.


# 3. L2 regularisation

**What is Regularization?**
One common form of regularization is called L2 regularization or weight decay, `which adds a penalty term to the loss function during training`. This penalty is proportional to the squared magnitude of the weights in the network. By adding this penalty term, the network is encouraged to have `smaller weights`, which helps in reducing overfitting.

![image.png](attachment:62cbc067-613f-43fb-bf8a-f88ecd03e55a.png)

- Assume that our regularization coefficient is so high that some of the weight matrices are nearly equal to zero.
![image.png](attachment:f996dec0-e764-474c-9e66-a35d61f1a8c2.png)

- This will result in a much simpler linear network and slight underfitting of the training data.

![image.png](attachment:4c941878-8710-4d2b-93a3-29de11dfc4c7.png)

- **Such a large value of the regularization coefficient is not that useful. We need to optimize the value of regularization coefficient in order to obtain a well-fitted model as shown in the image below.**
- 

# 4. Dropouts after convolutional layer, L2 in FC, and use BN after convolutional layer.




# 5. Add a new convolutional layer to the network. 
Note that by a ‘convolutional layer’, the professor refers to a convolutional unit with two sets of Conv2D layers with 128 filters each. The code for the additional convolution layer is given below:


# 6. Adding Feature Maps

In the previous experiment, you tried to increase the capacity of the model by adding a convolutional layer. Now, let’s try adding more feature maps to the same architecture.

Add more feature maps to the convolution layers: increase from 32 to 64 and 64 to 128. You can download the notebook below.


# Session Summary

In this session, you learnt to build and train CNNs in Keras and experimented with a few hyperparameters of the model. You also practised manually computing the number of parameters, output sizes, etc., of CNN-based architectures.


Based on these experiments, you saw that the performance of CNNs depends heavily on multiple hyperparameters: the number of layers, the number of feature maps in each layer, the use of dropouts, batch normalisation, etc. Thus, it is advisable to first fine-tune your model hyperparameters by conducting several experiments. Only when you are convinced that you have found the right set of hyperparameters should you train the model with a larger number of epochs (since the amount of time and computing power you have is usually limited).
