#Download pickle from https://drive.google.com/open?id=1HDFMrlLJu5bpa_Xrd7nbv8MbJUIVEvQM
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import sys
import os
from tensorflow.contrib import learn
import csv
from time import sleep
import pickle
import pandas as pd
from sklearn import model_selection, metrics

from keras.utils import to_categorical
from keras.layers import Lambda,merge,concatenate,Reshape,Bidirectional,Embedding,Dense, Input,MaxPooling2D,Conv2D,Dropout, Flatten, LSTM
from keras.initializers import Constant
from keras.models import Model, Sequential, load_model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import optimizers
from keras import backend as K
import keras
from keras.callbacks import ModelCheckpoint,Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.wrappers.scikit_learn import KerasClassifier

# Parameters
# ==================================================

np.random.seed(5)

params = {}
params["dev_sample_percentage"] = 0.2
params["embedding_dim"] = 300
params["filter_sizes"] = [3,4,5]
params["num_filters"] = 128
params["dropout_keep_prob"] = 0.5
params["l2_reg_lambda"] = 0.5
params["batch_size"] = 32
params["num_epochs"] = 100


print("loading data...")
x = pickle.load(open("./pickle.p","rb"))
revs, W, W2, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4], x[5]
print("data loaded!")# Load data

print('loading wgcca embeddings...')
wgcca_embeddings = np.load('user_gcca_embeddings.npz')
print('wgcca embeddings loaded')


ids = wgcca_embeddings['ids']
user_embeddings = wgcca_embeddings['G']
unknown_vector = np.random.normal(size=(1,100))
user_embeddings = np.concatenate((unknown_vector, user_embeddings), axis=0)
user_embeddings = user_embeddings.astype(dtype='float32')

wgcca_dict = {}
for i in range(len(ids)):
    wgcca_dict[ids[i]] = int(i)

csv_reader = csv.reader(open("discourse.csv"))
topic_embeddings = []
topic_ids = []
for line in csv_reader:
    topic_ids.append(line[0])
    topic_embeddings.append(line[1:])
topic_embeddings = np.asarray(topic_embeddings)
topic_embeddings_size = len(topic_embeddings[0])
topic_embeddings = topic_embeddings.astype(dtype='float32')
print("topic emb size: ",topic_embeddings_size)

topics_dict = {}
for i in range(len(topic_ids)):
    try:
        topics_dict[topic_ids[i]] = int(i)
    except TypeError:
        print(i)

word_idx_map["@"] = 0
rev_dict = {v: k for k, v in word_idx_map.items()}
max_l = 100

df = pd.DataFrame(revs)
train_x, test_x, train_y, test_y = model_selection.train_test_split(df.filter(['author','text','topic']), df['label'],test_size=params["dev_sample_percentage"])
train_y = to_categorical(train_y,2)
test_y = to_categorical(test_y,2)

def token_and_pad(text):
    seq = np.asarray([word_idx_map[word] for word in text.split()])
    if len(seq) < max_l:
        seq = np.append(seq,np.zeros(max_l-len(seq))).astype(int)
    else:
        seq = seq[0:max_l].astype(int)
    return seq

def author_name2id(author):
    try:
        return wgcca_dict['"'+str(author)+'"']
    except:
        return 0

def topic_name2id(topic):
    return topics_dict[str(topic)]

train_x['text'] = train_x['text'].map(lambda x: token_and_pad(x))
test_x['text'] = test_x['text'].map(lambda x: token_and_pad(x))
train_x['author'] = train_x['author'].map(lambda x: author_name2id(x))
test_x['author'] = test_x['author'].map(lambda x: author_name2id(x))
train_x['topic'] = train_x['topic'].map(lambda x: topic_name2id(x))
test_x['topic'] = test_x['topic'].map(lambda x: topic_name2id(x))

text = train_x['text'].values
author = train_x['author'].values
topic = train_x['topic'].values
X = []
for i in range(len(author)):
    row = list(text[i])
    row.append(author[i])
    row.append(topic[i])
    X.append(row)

text = test_x['text'].values
author = test_x['author'].values
topic = test_x['topic'].values
Xtest = []
for i in range(len(author)):
    row = list(text[i])
    row.append(author[i])
    row.append(topic[i])
    Xtest.append(row)

X = np.asarray(X)
Xtest = np.asarray(Xtest)

#https://github.com/keras-team/keras/issues/5400
def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def create_model():
    inp = Input(shape=(max_l+2,), dtype='int32')
    text_inp = Lambda(lambda x: x[:,:max_l], output_shape=(max_l,))(inp)
    user_inp = Lambda(lambda x: x[:,max_l], output_shape = (1,))(inp)
    topic_inp = Lambda(lambda x: x[:,max_l+1], output_shape = (1,))(inp)

    text_emb = Embedding(len(vocab)+1,params["embedding_dim"],embeddings_initializer=Constant(W),input_length=max_l,trainable=False)(text_inp)
    reshape_text = Reshape((max_l,params["embedding_dim"]) + (1, ), input_shape=(max_l,params["embedding_dim"]))(text_emb)
    conv1 = Conv2D(filters=params["num_filters"],kernel_size=(params["filter_sizes"][0],params["embedding_dim"]),strides=[1,1],padding="valid",activation="relu",use_bias=True,bias_initializer=Constant(0.1))(reshape_text)
    max1 = MaxPooling2D(pool_size=(max_l - params["filter_sizes"][0] + 1,1),strides=1,padding="valid")(conv1)
    conv2 = Conv2D(filters=params["num_filters"],kernel_size=(params["filter_sizes"][1],params["embedding_dim"]),strides=[1,1],padding="valid",activation="relu",use_bias=True,bias_initializer=Constant(0.1))(reshape_text)
    max2 = MaxPooling2D(pool_size=(max_l - params["filter_sizes"][1] + 1,1),strides=1,padding="valid")(conv2)
    conv3 = Conv2D(filters=params["num_filters"],kernel_size=(params["filter_sizes"][2],params["embedding_dim"]),strides=[1,1],padding="valid",activation="relu",use_bias=True,bias_initializer=Constant(0.1))(reshape_text)
    max3 = MaxPooling2D(pool_size=(max_l - params["filter_sizes"][2] + 1,1),strides=1,padding="valid")(conv3)

    num_filters_total = params["num_filters"] * len(params["filter_sizes"])
    h_pool = concatenate([max1,max2,max3],axis=3)
    h_flat = Reshape([num_filters_total])(h_pool)
    h_last = Dense(100)(h_flat)

    user_emb = Embedding(user_embeddings.shape[0],user_embeddings.shape[1],embeddings_initializer=Constant(user_embeddings),input_length=1,trainable=False)(user_inp)
    user_last = Reshape([-1])(user_emb)

    topic_emb = Embedding(topic_embeddings.shape[0],topic_embeddings.shape[1],embeddings_initializer=Constant(topic_embeddings),input_length=1,trainable=False)(topic_inp)
    topic_last= Reshape([-1])(topic_emb)

    combined = concatenate([h_last, user_last, topic_last], axis=1)
    drop = Dropout(params["dropout_keep_prob"])(combined)
    dense = Dense(400)(drop)
    out = Dense(2,kernel_regularizer=regularizers.l2(params["l2_reg_lambda"]))(dense)
    model = Model(inputs=inp, outputs=out)
    optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[f1_score,'accuracy'])
    return model

model = KerasClassifier(build_fn=create_model)
checkpoint = ModelCheckpoint('output/best.hdf5', monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
model.fit(X,train_y,validation_data=(Xtest,test_y),epochs=100,batch_size=64,verbose=1,callbacks=[checkpoint])


model = create_model()
model.load_weights("output/best.hdf5")
#model.save("best.h5s")

#Test Set
y_pred=model.predict(Xtest)
y_pred=y_pred.argmax(axis=-1)
Y_test=test_y.argmax(axis=-1)
print(metrics.f1_score(Y_test, y_pred, average='weighted'))
print(metrics.precision_score(Y_test, y_pred, average='weighted'))
print(metrics.recall_score(Y_test, y_pred, average='weighted'))

