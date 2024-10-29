# Handwritten-Digit-Recognition
Handwritten Digit Recognition
This project implements a Handwritten Digit Recognition system using Convolutional Neural Networks (CNN). The model is trained on the MNIST dataset, a popular dataset consisting of 28x28 grayscale images of handwritten digits ranging from 0 to 9.

Features
Train a CNN model using the MNIST dataset to recognize handwritten digits.
Use the trained model to predict digits from new images.
Save and load the trained model for future predictions.
Visualize the accuracy and loss of the training process.
Prerequisites
Ensure you have the following installed:

Python 3.x
TensorFlow or PyTorch (depending on the framework used)
Keras (if using TensorFlow/Keras)
Numpy
Matplotlib
OpenCV (optional for image pre-processing)
Dataset
The MNIST dataset is used for training and evaluating the model. It consists of 60,000 training images and 10,000 test images. Each image is 28x28 pixels and contains a single handwritten digit (0-9).

Installation
Clone the repository:
git clone https://github.com/kotha-karthik/handwritten-digit-recognition.git
cd handwritten-digit-recognition
Install the required dependencies:
pip install -r requirements.txt
Download the MNIST dataset (if not using built-in Keras/PyTorch MNIST dataset):
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

Training the Model
To train the CNN model:

Run the training script:
python train.py
The model will start training, and after completion, it will be saved as model.h5 or model.pth.

Testing the Model
To test the trained model on the test data:

Run the test script:
python test.py
You will see the accuracy and other evaluation metrics.

Predicting Digits
You can also predict digits on new images using the trained model:

Place your digit image in the input_images folder.

Run the prediction script:
python predict.py --image input_images/sample_digit.png
The predicted digit will be displayed.

Model Architecture
The Convolutional Neural Network (CNN) consists of the following layers:

Input Layer: 28x28 grayscale image
Conv2D Layer: 32 filters, 3x3 kernel size, ReLU activation
MaxPooling2D Layer: 2x2 pool size
Conv2D Layer: 64 filters, 3x3 kernel size, ReLU activation
MaxPooling2D Layer: 2x2 pool size
Flatten Layer
Dense Layer: 128 neurons, ReLU activation
Output Layer: 10 neurons (for digits 0-9), Softmax activation


Results
The model achieves an accuracy of 99% on the test set. It can efficiently recognize handwritten digits with high accuracy.
