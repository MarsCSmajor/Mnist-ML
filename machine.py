import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist # 28x28 image of hand written digits data set

(x_train, y_train), (x_test, y_test) = mnist.load_data() # load the data into 4 categories

#scale/normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)# normalize the data for x train and test
x_test = tf.keras.utils.normalize(x_test, axis=1) # organize/ normalize data
# scales it from 0 to 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # input layer

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))# hidden layers 
# 128 units/neurons per layer
#activation function: rectify linear (relu)

model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))# output layer

# 10 classifications
# activation function: softmax --> probability distribution

#parameters

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#optimizer: adam algorithm in ML
#loss: degree of error, what the model got wrong. --> sparse_categorical_crossentropy how loss is going to be calculated 
#metrics: overall accuracy of the model


#train model

model.fit(x_train,y_train,epochs=3) # epochs is the number of times in which the model retrains the data
#######################

val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss,val_acc)



# make a prediction
predictions = model.predict([x_test])

print(np.argmax(predictions[5])) #displays the predicted number

#print(x_train[0])

#plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()



plt.imshow(x_test[5])

plt.show()

