'''

Processes pre-processed data to train and feed into a RNN model 

'''

# IMPORTS (DATA PROCESSING)
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder


# GET PRE-PROCESSED DATA
df = pd.read_csv('processed_data_V2.csv')
df.tweet=df.tweet.astype(str) # make sure all are strings
# tweets = df[['tweet']].to_numpy().flatten()
# valence = df[['valence']].to_numpy().flatten()

# SPLIT 
    # 20-80 split
X_train, X_test, y_train, y_test = train_test_split(df.tweet, df.valence, test_size=0.4)


# TOKENIZE
    # to convert text to sequences of integers
    # Limit number of words to NB_WORDS most frequenct words
    # clean tweets with some filters, set to lowercase, split on spaces
tk = Tokenizer()

tk.fit_on_texts(X_train) 
X_train_tokenized = tk.texts_to_sequences(X_train)
X_test_tokenized = tk.texts_to_sequences(X_test)

# PAD 
X_train_tokenized_padded = pad_sequences(X_train_tokenized)
X_test_tokenized_padded = pad_sequences(X_test_tokenized)
NB_WORDS = len(tk.index_word) + 1 # Parameter indicating the number of words we'll put in the dictionary


# ENCODE target var (One Hot Encoding)

# Data to be converted
 
# Convert valence into numbers 

# le = LabelEncoder()
# y_train_le = le.fit_transform(y_train)
# y_test_le = le.transform(y_test)

# # # One hot encode
# y_train_OH = to_categorical(y_train_le)
# y_test_OH = to_categorical(y_test_le)

# RENAME FINAL TRAINING SET
X_train_FIN = X_train_tokenized_padded
# y_train_FIN  = y_train_OH 

# Using 1: pos; 0: neg
y_train_FIN = np.array([1 if y==4 else 0 for y in y_train])

# WORD EMBEDDING FOR TWEETS
    # use a pre-trained word embedding from glove database

    # LOAD glove database
GLOVE_DIM = 50  # Number of dimensions of the GloVe word embeddings

emb_dict = {}                      # create embedding dict
glove = open('glove.6B.50d.txt', 'rb')   # open file 

# Fill embedding dict
for line in glove:
    values = line.split()
    word = values[0].decode('UTF-8')
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector

glove.close()

# Look up word embedding for tweets in EMBEDDING DICT 
emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM)) # if embedding not found, 0
for w, i in tk.word_index.items():
    if i < NB_WORDS:
        vect = emb_dict.get(w)
        if vect is not None:
            emb_matrix[i] = vect
    else:
        continue

############################### END OF DATA PROC ##################################

# IMPORTS (BUILD RNN)
from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.layers.merge import Concatenate
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import metrics
import pickle
from keras import callbacks
from keras import models

# BUILD RN MODEL
model = Sequential()

# Add a pre-trained embedding layer 
    # converts word indices to glove word embedding vectors as theyre fed in

model.add(Embedding(NB_WORDS, GLOVE_DIM, weights=[emb_matrix],input_length=X_train_FIN.shape[1], trainable=False ))

# First LSTM layer 
    # return sequence so that we can feed into second LSTM layer
model.add(LSTM(128, return_sequences = True, activation='relu'))
model.add(Dropout(.2))

# # Second LSTM layer 
# model.add(LSTM(64, return_sequences = True, activation='relu'))
# model.add(Dropout(.2))

# Second LSTM layer 
# Don't return sequence this time, because we're feeding into a fully-connected layer
model.add(LSTM(128, activation='relu'))
model.add(Dropout(.2))

# Dense 1
model.add(Dense(32, activation='relu'))
model.add(Dropout(.2))

# Dense 2 (final vote)
model.add(Dense(1, activation = 'sigmoid'))


# # Loda model if saved 
# model = models.load_model("./model_save/model.80-0.48.h5")

# Print model summary
print(model.summary())

# Make
LOSS = 'binary_crossentropy' # Binary categorical y 
OPTIMIZER = 'ADAM' # RMSprop tends to work well for recurrent models
model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = [metrics.binary_accuracy])

# TRAIN MODEL 
EPOCHS = 100
BATCH_SIZE = 512
TEST_SIZE = 0.05
MY_CALLBACKS = [callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.0002),
                callbacks.ModelCheckpoint(filepath='model_data/model.{epoch:02d}-{val_loss:.2f}.h5')]

model.fit(X_train_FIN, y_train_FIN, epochs=EPOCHS, batch_size=BATCH_SIZE,  callbacks=MY_CALLBACKS, validation_split=TEST_SIZE)

# EVALUATE 
scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

