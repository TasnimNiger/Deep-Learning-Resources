#**Understanding and Implementing the Activation Function**

##**What are Activation Functions?**
Activation functions are a critical component of neural networks that introduce non-linearity into the model, allowing networks to learn complex patterns and relationships in the data. These functions play an important role in the hyperparameters of AI-based models. 

There are numerous different activation functions to choose from. For data scientists and machine learning engineers, knowing which function or series of functions to train a neural network can be challenging. 

In a neural network, the weighted sum of inputs is passed through the activation function.

Y = Activation function(∑ (weights*input + bias))

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/ff9d94a2-3435-4431-823e-e7c913538b8c)

##**Why Neural Networks Need Activation Functions?**
Activation functions are used to remove the linearity from the neural network. If we do not apply an activation function then the output signal would simply be a simple linear function. In other words, it wouldn’t be able to handle large volumes of complex data. Activation functions are an additional step in each forward propagation layer but a valuable one. 


Activation functions are of two types based on how it is used in an ML model.

Activation functions that are used in output layers of ML models (think classification problems). The primary purpose of these activation functions is to squash the value between a bounded range like 0 to 1.
Activation functions that are used in hidden layers of neural networks. The primary purpose of these activation functions is to provide non-linearity without which neural networks cannot model non-linear relationships.

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/f1a54291-a281-4a71-9db4-ec5e75fc0033)

##**Comparison Between Linear Activation Function and Other Activation Functions**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/978a0327-28c9-4afa-817f-365e918c6a59)
