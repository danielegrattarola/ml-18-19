# Assignment 2

In this assignment you have to solve **one** exercise of your choice between the two that we propose:

1. Implement a neural network to classify images following our instructions on what architecture and which learning procedure to adopt.
2. Implement an **autoencoder** architecture to process images in an _unsupervised_ way, and generate new sample images.

Exercise 1 is very similar to what we saw during the labs, but requires you to follow **exactly** our specifications.   
Exercise 2 asks you to implement an architecture that we have not seen in class, and requires you to do a bit of individual work. However, it is also more interesting, and we believe you will have more fun.

Once completed, you have to submit your solution as a zip file on the iCorsi platform. 

---

## Exercise 1

Implement a multi-class classifier to identify images of handwritten digits.

### Tasks

1. Build a neural network with the following architecture: 
    - Convolutional layer, with 32 filters of size 3 by 3, stride of 1 by 1, and ReLU activation;
    - Max pooling layer, with pooling size of 2 by 2;
    - Dropout layer with dropout probability of 0.3;
    - Convolutional layer, with 64 filters of size 3 by 3, stride of 2 by 2, and ReLU activation;
    - Layer to convert the 2D feature maps to flat vectors;
    - Dropout layer with dropout probability of 0.3;
    - Dense layer with 128 neurons and ReLU activation;
    - Dropout layer with dropout probability of 0.3;
    - Dense output layer with softmax activation;
2. Bonus: add L2 regularization with factor 0.0005 on the weights to every Dense layer, use Leaky ReLU activations instead of ReLU ones. Set the slope of the Leaky ReLU for `x < 0` to 0.15.
3. Explain if using a Softmax as last activation is appropriate or not.
4. Train the model:
    - Use the Adam optimization algorithm, with a learning rate of 0.002 and a batch size of 64;
    - Implement early stopping, monitoring the **validation accuracy** of the model with a patience of 10 epochs and use 20% of the data as validation set (Hint: see the `EarlyStopping` and `ModelCheckpoint` callbacks of Keras. `ModelCheckpoint` allows you to save the model to a file every time the validation accuracy improves);
    - When early stopping kicks in, restore the best model found during training using `model.load_weights(filename)`;
    - Explain what is early stopping, why it is useful and how you implemented with Keras. (Hint: show us that you understood what is it and how your code works.) 
5. Assess the performance of the network using 10-fold cross-validation, and provide a fair estimation of the misclassification error that you expect on new and unseen images. Does your estimate correspond to the actual test performance? Why?

In order to complete the assignment, you must submit a zip file on the iCorsi platform containing: 

1. a PDF file describing how you solved the assignment, covering all the points described above (at most 2500 words, no code!);
2. a working example of how to load your **trained model** from file;
3. the source code you used to build, train, and evaluate your model.


### Evaluation criteria

You will get a positive evaluation if:

- you demonstrate a clear understanding of the main tasks and concepts;
- you provide a clear description of your solution;
- the performance assessment is conducted appropriately;
- your code runs out of the box (i.e., without us needing to change your code to evaluate the assignment);
- your code is properly commented;

You will get a negative evaluation if: 

- we realise that you copied your solution;
- the description of your solution is not clear, or it is superficial;
- your code requires us to edit things manually in order to work;
- your code is not properly commented;

---

## Exercise 2
Autoencoders (AEs) are a class of **unsupervised** machine learning models that are used to learn a representation of data, without the need for labels.
In practice, AEs are usually implemented as neural networks that take some data X as input, and try to replicate the very same data X as output.  
An AE is composed of two main blocks: an encoder that maps input data X to a latent representation Z, and a decoder which maps the latent representation Z back to the original domain.
```
X -> Z = Enc(X) -> X_rec = Dec(Z)
```
You can find an simple introduction on [wikipedia](https://en.wikipedia.org/wiki/Autoencoder).

Implement an autoencoder using a neural network and train it on the entire MNIST dataset (join the training and test sets). Notice that it is an unsupervised approach, and no labels are needed at this level.
You can choose to use only dense layers or, if you think it is more appropriate, use different ones, like Conv2D and Conv2DTransposed (always check the documentation and, possibly, online tutorials).   
Once you are done and satisfied of your auto-encoder, it is useful to save the model to file, so that you don't need retrain at every session. It is also recommended to set a seed for the pseudo-random number generator, so that anyone else can replicate your results, and we are able to check your work in case of problems.

### Tasks
1. Report some relevant applications of autoencoders. 
2. What does the latent space look like? Make a _scatter_ plot of the training set mapped to the latent space, using different colours according to which class they belong;
3. By sampling random points in the latent space and passing them as input to the decoder, your are able to generate new and unseen images. If you did everything right, the generated digits will be understandable by a human. Plot a few of them for each digit. (hint: the next bonus task might help you in deciding where to sample in the latent space, though it is not necessary).
4. Bonus: build a classifier for the images to identify which digit was present in the image. The classifier should receives as input the latent-space representation of the images. You are allowed to pick any classifier of you choice. Do you think it is better to train a classifier directly on the images or, as we asked you to do, operating on the latent representation?

In order to complete the assignment, you must submit a zip file on the iCorsi platform containing: 

1. a PDF file describing how you solved the assignment, covering all the points described above (at most 2500 words, no code!);
2. a working example of how to load your **trained model** from file;
3. the source code you used to build, train, and evaluate your model.


### Evaluation criteria

You will get a positive evaluation if:

- you demonstrate a clear understanding of how the concepts presented in class apply here;
- you provide a clear description of your solution;
- you provide reasonable answers to the questions;
- your code is properly commented;

You will get a negative evaluation if: 

- we realise that you copied your solution;
- we realise that important notions from those presented in the course are missing;
- the description of your solution is not clear, or it is superficial;
- your code requires us to edit things manually in order to work;
- your code is not properly commented;

---

## Dataset

Both exercises use the MNIST dataset of handwritten digits, available freely online. 

To load the data, you can use the `datasets` submodule of Keras: 

```py
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test)) = mnist.load_data()
```

## Tools

Your solution must be entirely coded in **Python 3** ([not Python 2](https://python3statement.org/)), using the tools we have seen in the labs.
These include: 

- Numpy
- Scikit-learn
- Keras

The neural networks, in particular, must be coded in Keras. We recommend that you read the documentation of Keras **thoroughly** because all the required tasks can be completed using only Keras. On the [documentation page](https://keras.io) there is a useful search field that allows you to smoothly find what you are looking for. 

You can develop your code in Colab, like we saw in the labs, or you can install the libraries on your machine and develop locally.  
If you choose to work in Colab, you can then export the code to a `.py` file by clicking "File > Download .py" in the top menu.  
If you want to work locally, instead, you can install Python libraries using something like the [Pip](https://pypi.org/project/pip/) package manager. There are plenty of tutorials online. 
