## Task 3. Apply the code on 20_newsgroup data set we worked in the previous classes

# importing the required libraries
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# importing 20_newsgroup data set
from sklearn.datasets import fetch_20newsgroups
# selecting the required categories of data
catgs = ['alt.atheism', 'sci.space']
# loading the categories into Datafram
twenty_ng_train = fetch_20newsgroups(subset='train', shuffle=True, categories=catgs)

# extracting the features and target
sentences = twenty_ng_train.data
y = twenty_ng_train.target

# tokenizing the data
tokenizer = Tokenizer(num_words=2000)

# preprocessing the data
from keras.preprocessing.sequence import pad_sequences
max_review_len = max([len(s.split()) for s in sentences])
vocab_size = len(tokenizer.word_index)+1
sentencesPre = tokenizer.texts_to_sequences(sentences)
padded_docs = pad_sequences(sentencesPre, maxlen=max_review_len)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

# implementing the model
from keras.layers import Embedding, Flatten
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu',input_dim=max_review_len))
model.add(layers.Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

history3 = model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

# evaluating the model
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data: Loss = {}, accuracy = {}".format(test_loss, test_acc))

# predicting the Value for test sample
predt = model.predict_classes(X_test[[2],:])
print("Actual Prediction", y_test[1], "Predicted Prediction", predt)

# loss and accuracy plot
import matplotlib.pyplot as mplt
# history summarization for accuracy plot
mplt.plot(history3.history['acc'])
mplt.plot(history3.history['val_acc'])
mplt.title('model accuracy')
mplt.ylabel('accuracy')
mplt.xlabel('epoch')
mplt.legend(['accuracy', 'val_accuracy'], loc='upper left')
mplt.show()

# history summarization for loss plot
mplt.plot(history3.history['loss'])
mplt.plot(history3.history['val_loss'])
mplt.title('model loss')
mplt.ylabel('loss')
mplt.xlabel('epoch')
mplt.legend(['loss', 'val_loss'], loc='upper right')
mplt.show()



