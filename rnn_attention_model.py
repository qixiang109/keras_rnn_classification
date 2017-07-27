# *-encoding=utf-8
import keras
from keras.layers import Input, Dense, GRU
from keras.models import Model,Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import backend as K
import cPickle as pickle
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class RNNAttentionModel(object):
    
    def __init__(self,vocab_size=4000,hidden_size=128, num_layers=2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._build_graph()
        
    def _build_graph(self):
        model = Sequential()
        
        #embedding
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size))
        
        #gru
        for i in xrange(self.num_layers):
            if i<self.num_layers-1:
                model.add(GRU(units=self.hidden_size,activation='relu',return_sequences=True,implementation=2))
            else:
                model.add(GRU(units=self.hidden_size,activation='relu',return_sequences=False,implementation=2))
        
        #softmaxt
        model.add(Dense(1, activation='sigmoid'))
        
        # PFA, prob false alert for binary classifier
        def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
            y_pred = K.cast(y_pred >= threshold, 'float32')
            # N = total number of negative labels
            N = K.sum(1 - y_true)
            # FP = total number of false alerts, alerts from the negative class labels
            FP = K.sum(y_pred - y_pred * y_true)    
            return FP/N
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        # P_TA prob true alerts for binary classifier
        def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
            y_pred = K.cast(y_pred >= threshold, 'float32')
            # P = total number of positive labels
            P = K.sum(y_true)
            # TP = total number of correct alerts, alerts from the positive class labels
            TP = K.sum(y_pred * y_true)    
            return TP/P
        
        def auc(y_true, y_pred):   
            ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
            pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
            pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
            binSizes = -(pfas[1:]-pfas[:-1])
            s = ptas*binSizes
            return K.sum(s, axis=0)
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
        print(model.summary())
        self.model=model
    
    def fit(self,X_train,y_train,X_test, y_test):
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

    
    def save(self, path):
        pass
    
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = np.load('data/data.npy')
    X_train = sequence.pad_sequences(X_train, maxlen=200)
    X_test = sequence.pad_sequences(X_test, maxlen=200)    
    
    model = RNNAttentionModel(vocab_size=4000,hidden_size=128, num_layers=2)
    model.fit(X_train, y_train, X_test, y_test)
    
    
    