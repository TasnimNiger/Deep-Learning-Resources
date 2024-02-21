# **Understanding and Implementing the Activation Function**

## **What are Activation Functions?**
Activation functions are a critical component of neural networks that introduce non-linearity into the model, allowing networks to learn complex patterns and relationships in the data. These functions play an important role in the hyperparameters of AI-based models. 

There are numerous different activation functions to choose from. Knowing which function or series of functions to train a neural network can be challenging for data scientists and machine learning engineers. 

In a neural network, the weighted sum of inputs is passed through the activation function.

Y = Activation function(∑ (weights*input + bias))

<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/ff9d94a2-3435-4431-823e-e7c913538b8c" width="700" height="500" align="center">

## **Why Neural Networks Need Activation Functions?**
*  Activation functions are used to remove the linearity from the neural network. If we do not apply an activation function, the output signal would simply be a simple linear function. In other words, it wouldn’t be able to handle large volumes of complex data. Activation functions are an additional step in each forward propagation layer but valuable. 

*  Activation functions used in ML models' output layers (think classification problems). The primary purpose of these activation functions is to squash the value between a bounded range like 0 to 1.
  <img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/80737c0e-3b75-4c1f-9f3d-bbb34c94b230" width="600" height="400" align="center">

* Activation functions that are used in hidden layers of neural networks. The primary purpose of these activation functions is to provide non-linearity, without which neural networks cannot model non-linear relationships.
  
*  During backpropagation, gradients calculated for each layer depend on the derivative of the activation function.
  
*  The choice of activation function affects the overall training speed, stability, and convergence of neural networks.


### **Activation functions are mainly of two types based on their use in an ML model.**
## **Linear Activation Functions**
**Equation : f(Z) = Z**

**Function Input Range:** (- ∞,∞)

**Function Output Range:** (- ∞,∞)

**Code Snippet:**
def linear(z) :
  fn = z
  return(fn)

  **Graph:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/6c3db64d-d511-4e71-84e5-7ecea92203ee)


## **Non-Linear Activation Functions**

### **a) Sigmoid Activation Functions**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/bdfa870d-bbab-480e-b2ee-7d955f464c05)

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (0,1)

**Code Snippet:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/0c5359cc-0b9f-4b10-a506-7eaac94cc6db)

**Graph:**

**Key features:**

*  This is also called the logistic function used in logistic regression models.
*  The sigmoid function has an s-shaped graph.
*  Clearly, this is a non-linear function.
*  The sigmoid function converts its input into a probability value between 0 and 1.
*  It converts large negative values towards 0 and large positive values towards 1.
*  It returns 0.5 for the input 0. The value 0.5 is known as the threshold value, which can decide which of two classes a given input belongs to.

**Usage:**

*  In the early days, the sigmoid function was used to activate the hidden layers in MLPs, CNNs and RNNs.
*  However, the sigmoid function is still used in RNNs.
*  We do not usually use the sigmoid function for the hidden layers in MLPs and CNNs. Instead, we use ReLU or Leaky ReLU there.
*  The sigmoid function must be used in the output layer when we build a binary classifier in which the output is interpreted as a class label depending on the probability value of input returned by the function.
*  The sigmoid function is used when we build a multilabel classification model in which each mutually inclusive class has two outcomes. Do not confuse this with a multiclass classification model.
  
### **b) Tanh Activation Functions**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/df470087-bfff-4fba-b6c8-2e0a91fae527)

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-1,1)

**Graph:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/1aabd856-1d9a-42a4-a927-f05256d9f653)

**Code Snippet:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/2c6e6598-54fb-407f-8aab-0221ec98e61d)

**Key features:**

*  The output of the tanh (tangent hyperbolic) function always ranges between -1 and +1.
*  Like the sigmoid function, it has an s-shaped graph. This is also a non-linear function.
*  One advantage of using the tanh function over the sigmoid function is that the tanh function is zero centered. This makes the optimization process much easier.
*  The tanh function has a steeper gradient than the sigmoid function has.

**Usage:**

Until recently, the tanh function was used as an activation function for the hidden layers in MLPs, CNNs and RNNs.
However, the tanh function is still used in RNNs.
Currently, we do not usually use the tanh function for the hidden layers in MLPs and CNNs. Instead, we use ReLU or Leaky ReLU there.
We never use the tanh function in the output layer.

### **c) ReLU Activation Functions**

**Equation: f(Z) = max(0,Z)**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (0,∞)

**Graph:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/de2ae401-3a97-4c88-8d37-172e511c9016)

**Code Snippet:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/dc9b687c-eccc-4686-82e8-8f64fa43faaf)



### **d) Leaky Relu**

**Equation: f(Z) = max(0.01Z,Z)**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-∞,∞)

**Graph:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/d3510d18-f5f5-474e-8cff-e1597bd07803)

**Code Snippet:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/30cd20f1-d6ec-45b2-84bb-491364d309ad)


###  **e) Parametric ReLU(PReLU)**

**Equation: f(Z) = max(αZ,Z)**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-∞,∞)

**Graph:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/9e955d70-90fc-45d0-b05e-b0bf09d89976)

**Code Snippet:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/22afaefc-d0cf-4de0-afd8-b1ed870dff12)

****
###  **f)Exponential Linear Unit(ELU)**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/489e243a-753d-416a-b574-208ee40e01ed)

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-α,∞)

**Graph:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/2a95d39a-ed13-475c-8822-37f7ef7e1f36)

**Code Snippet:**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/03d0b5da-c16d-481b-8ab9-f5cb1ba5528c)


###  **g) Softmax Activation Functions**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (0,1)

**Graph:**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/36d345fe-b51e-41e1-bcf2-3c3eebe0a15b)

**Code Snippet:**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/423e52f4-d8ad-4cd1-a668-5a5d1f1d238e)



## **Choosing the right Activation Function**

<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/8c91052c-3e28-4990-8346-e98a1a31b59a" width="600" height="400" align="center">

Now that we have seen so many activation  functions, we need some logic/heuristics to know which activation function should be used in which situation. Good or bad – there is no rule of thumb.

However, depending on the properties of the problem, we can make a better choice for easy and quicker network convergence.

*  Sigmoid functions and their combinations generally work better in the case of classifiers
*  Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
*  ReLU function is a general activation function and is used in most cases these days
*  If we encounter a case of dead neurons in our networks, the leaky ReLU function is the best choice
*  Always keep in mind that the ReLU function should only be used in the hidden layers
*  As a rule of thumb, you can begin with using the ReLU function and then move over to other activation functions in case ReLU doesn’t provide optimum results. 


## **Advantages and Disadvantages**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/9609a39e-ca9b-41bf-bc82-8eb3b967255b)

##  **Vanishing Gradients:**

*  Vanishing gradients occur when the derivatives of activation functions become extremely small, causing slow convergence or stagnation in training.
*  Sigmoid and tanh activation functions are known for causing vanishing gradients, especially in deep networks.

##  **Mitigating the Vanishing Gradient Problem:**

*  Rectified Linear Unit (ReLU) and its variants, such as Leaky ReLU, address the vanishing gradient problem by providing a non-zero gradient for positive inputs.
*  ReLU functions result in faster convergence due to the lack of vanishing gradients when inputs are positive.

##  **Role of Zero-Centered Activation Functions:**

*  Activation functions like ELU, which offer zero-centered output, help mitigate the vanishing gradient problem by providing both positive and negative gradients.
*  Zero-centered functions contribute to stable weight updates and optimization during training.

## **Quick Summary of different activation functions**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/978a0327-28c9-4afa-817f-365e918c6a59)

## **References:**
1. [Activation functions in Neural Networks](https://medium.com/@datta.nagraj/activation-functions-in-neural-networks-6ffe7b723420)
2. [Linear Activation Function](https://iq.opengenus.org/linear-activation-function/)
3. [Unlocking The Power of Activation Functions in Neural Networks](https://www.analyticsvidhya.com/blog/2023/10/activation-functions-in-neural-networks/)
4. [Neural Networks and Activation Function](https://www.analyticsvidhya.com/blog/2021/04/neural-networks-and-activation-function/)
5. [Introductory Guide on the Activation Functions](https://www.analyticsvidhya.com/blog/2022/03/introductory-guide-on-the-activation-functions/)
