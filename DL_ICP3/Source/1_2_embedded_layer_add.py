# 1. In the code provided, there are three mistake which stop the code to get run successfully; find those mistakes and explain why they need to be corrected to be able to get the code run
# 2. Add embedding layer to the model, did you experience any improvement?

# Task 1
# importing the required libraries
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# reading the data file
df = pd.read_csv('imdb_master.csv', encoding='latin-1')
print(df.head())

# extracting the features and target
sentences = df['review'].values
y = df['label'].values

# tokenizing the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)

# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

# label encoding the target and splitting the data
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

input_dim = np.prod(X_train.shape[1:])
print(input_dim)

# implementing the model
model = Sequential()
model.add(layers.Dense(300, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# evaluating the model
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# loss and accuracy plot
import matplotlib.pyplot as mplt
# summarize history for accuracy
mplt.plot(history.history['acc'])
mplt.plot(history.history['val_acc'])
mplt.title('model accuracy')
mplt.ylabel('accuracy')
mplt.xlabel('epoch')
mplt.legend(['accuracy', 'val_accuracy'], loc='upper left')
mplt.show()

# summarize history for loss
mplt.plot(history.history['loss'])
mplt.plot(history.history['val_loss'])
mplt.title('model loss')
mplt.ylabel('loss')
mplt.xlabel('epoch')
mplt.legend(['loss', 'val_loss'], loc='upper right')
mplt.show()

# Task 2
# Adding embedding layer to the model

# embedding Layer pre-processing
from keras.preprocessing.sequence import pad_sequences
pureSentences = df['review'].values
max_review_len = max([len(s.split()) for s in pureSentences])
vocab_size = len(tokenizer.word_index)+1
sentencesPre = tokenizer.texts_to_sequences(pureSentences)
padded_docs = pad_sequences(sentencesPre, maxlen=max_review_len)

X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)
print(vocab_size)
print(max_review_len)

# implementing the model by adding the embedding layer to the model
from keras.layers import Embedding, Flatten
m = Sequential()
m.add(Embedding(vocab_size, 50, input_length=max_review_len))
m.add(Flatten())
m.add(layers.Dense(300, activation='relu',input_dim=max_review_len))
m.add(layers.Dense(3, activation='softmax'))
m.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])
history2 = m.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# evaluating the model
[test_loss1, test_acc1] = m.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss1, test_acc1))

# loss and accuracy plot after adding the embedding layer
# history summarization for accuracy plot
mplt.plot(history2.history['acc'])
mplt.plot(history2.history['val_acc'])
mplt.title('model accuracy')
mplt.ylabel('accuracy')
mplt.xlabel('epoch')
mplt.legend(['accuracy', 'val_accuracy'], loc='upper left')
mplt.show()

# history summarization for loss plot
mplt.plot(history2.history['loss'])
mplt.plot(history2.history['val_loss'])
mplt.title('model loss')
mplt.ylabel('loss')
mplt.xlabel('epoch')
mplt.legend(['loss', 'val_loss'], loc='upper right')
mplt.show()

predt = m.predict_classes(X_test[[2],:])
print("Actual Prediction", y_test[1], "Predicted Prediction", predt)