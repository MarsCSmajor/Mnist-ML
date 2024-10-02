Using Mnist data set in which contains 28x28 images of hand drawn numbers from 0 to 9, we make a model to train on the data and make a prediction.

I use tensorflow to create/train the model. 
We normalize the data for training and testing data
we use keras as part of the tensorflow library to have an input layer, dense hidden layer of 128 neurons per layer and I use relu rectify linear as the activation

For the output layer uses 10 classifcations through softmax as the output layer

When compiling the model using the adam optimizer produces a higher accuracy when training the data. 

Other optimizers for model compilation:
Gradient Descent.
SGD.
AdaGrad.
RMSprop.
Adadelta.
AdaMax.
NAdam.
adam is commonly used due to being more efficient and having a higher accuracy value.

You can edit line 27 in the optimizer assignment and set it to whatever prefer optimizer. You will notice that the change in accuracy changes depending on which one do you use.



How to Compile code. 
Make sure to have the latest python version and pip/pip3 installed

have matplotlib install -->link: https://pypi.org/project/matplotlib/
have tensorflow install -->link: https://www.tensorflow.org/install
