# 1. Plot the loss and accuracy for both training data and validation data using the history object in the source code.
# 2. Plot one of the images in the test data, and then do inferencing to check what is the prediction of the model on that single image.
# 3. We had used 2 hidden layers and Relu activation.
# Try to change the number of hidden layer and the activation to tanh or sigmoid and see what happens.

# importing the libraries
from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as mt_plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape[1:])
# process the data
# convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
print(dimData)
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

# convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')

# scale data
train_data /=255.0
test_data /=255.0

# change the labels from integer to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=5, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

## 1
# plotting the loss using the history object

mt_plt.plot(history.history['loss'])                 # loss for training data
mt_plt.plot(history.history['val_loss'])             # loss for validation data
mt_plt.title('Model Loss')
mt_plt.ylabel('loss')
mt_plt.xlabel('epoch')
mt_plt.legend(['train', 'test'], loc='upper right')
mt_plt.show()

# plotting the accuracy using the history object

mt_plt.plot(history.history['accuracy'])             # accuracy for training data
mt_plt.plot(history.history['val_accuracy'])         # accuracy for validation data
mt_plt.title('Model Accuracy')
mt_plt.ylabel('accuracy')
mt_plt.xlabel('epoch')
mt_plt.legend(['train', 'test'], loc='upper left')
mt_plt.show()

## 2. fetching one of the images in the test data - plotting and predicting the model on that single image

# prediction of the model on the single image
test_img_pred = model.predict_classes(test_data[[23], :])               # index 23 - zero indexing (represents image 24)
print("\n The predicted single image in the test data is: ", test_img_pred)

# plotting one of the images in the test data
mt_plt.imshow(test_images[23,:,:].reshape(28, 28)) #cmap='gray'
mt_plt.title('Display of single image in test data')
mt_plt.show()

## 3. change the number of hidden layer and the activation to tanh or sigmoid and see what happens.

model_1 = Sequential()
model_1.add(Dense(512, activation='tanh', input_shape=(dimData,)))
model_1.add(Dense(550, activation='tanh'))
model_1.add(Dense(576, activation='tanh'))
model_1.add(Dense(640, activation='tanh'))
model_1.add(Dense(10, activation='sigmoid'))

model_1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history_1 = model_1.fit(train_data, train_labels_one_hot, batch_size=256, epochs=15, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss_1, test_acc_1] = model_1.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss_1, test_acc_1))
