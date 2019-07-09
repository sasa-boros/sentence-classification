import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import keras.backend as K
import numpy as np
import random
from sklearn import metrics

def encode(labelEncoder, labels):
    enc = labelEncoder.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(labelEncoder, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return labelEncoder.inverse_transform(dec)
    
url = "https://tfhub.dev/google/elmo/2"
elmoEmbedd = hub.Module(url)

data = pd.read_csv('spam.csv', encoding='latin-1')

# classes
y = list(data['v1'])
# sentences
x = list(data['v2'])

labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(y)

y_encoded = encode(labelEncoder, y)

x_train = np.asarray(x[:5000])
y_train = np.asarray(y_encoded[:5000])

x_test = np.asarray(x[5000:])
y_test = np.asarray(y_encoded[5000:])

def embed(sentence):
    return elmoEmbedd(tf.squeeze(tf.cast(sentence, tf.string)), signature="default", as_dict=True)["default"]

input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(embed, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(2, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Uncomment to retrain model
"""
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = model.fit(x_train, y_train, epochs=1, batch_size=32)
    model.save_weights('./elmo-model.h5')
"""

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo-model.h5')  
    predicts = model.predict(x_test, batch_size=32)

y_test = decode(labelEncoder, y_test)
y_preds = decode(labelEncoder, predicts)

print(metrics.confusion_matrix(y_test, y_preds))
print(metrics.classification_report(y_test, y_preds))