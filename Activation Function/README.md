# **Understanding and Implementing the Activation Function**

## **What are Activation Functions?**
Activation functions are a critical component of neural networks that introduce non-linearity into the model, allowing networks to learn complex patterns and relationships in the data. These functions play an important role in the hyperparameters of AI-based models. 

There are numerous different activation functions to choose from. Knowing which function or series of functions to train a neural network can be challenging for data scientists and machine learning engineers. 

In a neural network, the weighted sum of inputs is passed through the activation function.

Y = Activation function(∑ (weights*input + bias))

<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/ff9d94a2-3435-4431-823e-e7c913538b8c" width="700" height="500" align="center">

## **Why Neural Networks Need Activation Functions?**
Activation functions are used to remove the linearity from the neural network. If we do not apply an activation function, the output signal would simply be a simple linear function. In other words, it wouldn’t be able to handle large volumes of complex data. Activation functions are an additional step in each forward propagation layer but valuable. 

Activation functions used in ML models' output layers (think classification problems). The primary purpose of these activation functions is to squash the value between a bounded range like 0 to 1.
Activation functions that are used in hidden layers of neural networks. The primary purpose of these activation functions is to provide non-linearity, without which neural networks cannot model non-linear relationships.

<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/80737c0e-3b75-4c1f-9f3d-bbb34c94b230" width="600" height="400" align="center">

### **Activation functions are mainly of two types based on their use in an ML model.**
<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/8c91052c-3e28-4990-8346-e98a1a31b59a" width="600" height="400" align="center">



## **Advantages and Disadvantages**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/9609a39e-ca9b-41bf-bc82-8eb3b967255b)

## **Quick Summary of different activation functions**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/978a0327-28c9-4afa-817f-365e918c6a59)

## **References:**
1. [Activation functions in Neural Networks](https://medium.com/@datta.nagraj/activation-functions-in-neural-networks-6ffe7b723420)
2. [Linear Activation Function](https://iq.opengenus.org/linear-activation-function/)
