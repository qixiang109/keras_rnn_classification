# *-encoding=utf-8
from keras.datasets import imdb
import numpy as np


    
if __name__ == '__main__':
    np.save('data/data.npy',[1,2])
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                          num_words=4000,
                                                          skip_top=0,
                                                          maxlen=200,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3
                                                          )
    np.save('data/data.npy',[x_train,y_train,x_test,y_test])
    
    
    
    