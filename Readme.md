# Classification Using Artificial Neural Network

Simple 3 layer artificial neural network for classification of iris plants.

## Motivation
The motivation for this project was practice for skills learned in basic deep learning course.
Apart for implementation of ANN model, some basic data pre-processing techniques, such as
normalization and one-hot encoding was used. It also was aimed for improving my mathematical understanding
of forward run of Neural Network, Back-propagation, Activation Functions and Hyper-parameters.

## Data
Data is from Open source "Iris Data Set" from UCI, Machine Learning Repository. But I used the version available on [Kaggle Link](https://www.kaggle.com/willvegapunk/iris-data-set/data). I used this version as data was in *.csv format rather then
*.data format on UCI repository. I found it easy for *.csv to use.

### Important Information
Please keep downloaded data in the same folder as python files. 

## Framework
As this project was for learning purpose, I have written code in both in simple Numpy, and Tensorflow framework.
Tensorflow code was very low level, with just simple implementation of ANN.

## Folder Structure
The repository contains,

1. Iris.csv
    Contains data
2. iris.names
    Contains information about data set. i.e. its attributes, publications etc
3. main.py
    Contains models using numpy
4. main_tf.py
    Contains models using Tensorflow
5. utils.py
    Contains utility functions e.g. get_data, sigmoid, softmax

## Code
1. Model developed using Numpy
    The model is developed using class definition with name ANN, the class as function,
        fit: to fit model to data
        forward: for forward loop
        predict: to find prediction y given x
        score: to find error of model for certain data.
2. Model developed using TensorFlow
    The model is developed using class definition with name ANN_TF, the class is based on simple tensorflow
    structure.

## Usage
To run the simple numpy file
'''
python main.py
'''
To run the Tensorflow based file
'''
python main_tf.py
'''

## Results
Results from model ANN are better as we can use regularization techniques for this model. but for model
ANN_TF the result is not so good. ANN gives accuracy ~= 0.9 and ANN_TF ~= 0.8. The learning rate can be given as
input to model to check its response.

## Credits
I would like to give credit to [lazyProgrammer](https://lazyprogrammer.me) for developing a very comprehensive and understandable course for deep learning. Also UCI for making the Iris Data Set available. Also [Azeem Bootwala](https://github.com/azeembootwala) for introduction to this data set
and encouragement.

## License
MIT :copyright: [Jalil Ahmed](https://www.linkedin.com/in/jalil-siddiqui/).
