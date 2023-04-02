# TwoLayerNet
Neural Network and Deep Learning Homework 1.

# Running Experiments
To run this program and train the model entirely, run ```python main.py```. (The current hyperparameters of the model have been set up.)

To test the model parameter, download file ```model_parameters.pkl``` from https://pan.baidu.com/s/1fxfUzVhtnzkoz8H3ZElazQ?pwd=pimw (password: pimw) and put it in the same directory as ```main.py```, run ```test.py``` to get the accuracy on test dataset of the pre-trained model.

If you have trouble reproducing any results, please raise a GitHub issue with your logs and results. Otherwise, if you have success, please share your trained model weights with us and with the broader community!

# Installation
### Install packages
* **NumPy**
* **pickle**
* **collections**
* **Matplotlib**

# Code Structure
* **```main.py``` run the program, save the model**
* **```train.py``` train the model by mini-batch SGD, and realize learning rate decay (Inverse Decay). Show the train loss and test loss as well**
* **```layers.py``` implement the required layers in ```model.py```**
* **```functions.py``` implement the loss function and softmax operation**
* **```mnist.py``` download the MNIST Dataset by load_mnist() function (saved as "mnist.pkl")**
* **```test.py``` reload the parameters of model and print the accuracy on the test dataset**

# Supplement
The program downloads the MNIST Dataset at the first time it runs, and then directly uses the local dataset. The final result is saved as a ```result.jpg``` image and the model parameters are saved as the ```model_parameters.pkl``` file.
