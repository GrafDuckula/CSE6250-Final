from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
# from tflearn.data_preprocessing import ImagePreprocessing
# from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import pickle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def main():
    print("Using NN to train")
    

def read_train_data():
    X_train_pickle = open("data.mat","rb")
    Y_train_pickle = open("labels.mat","rb")
    X_train = pickle.load(X_train_pickle)
    Y_train = pickle.load(Y_train_pickle)
    X_train_pickle.close()
    Y_train_pickle.close()
    return X_train, Y_train

X_train, Y_train = read_train_data()
print(X_train.shape)
print(Y_train[0])


train_test_split = np.random.rand(X_train.shape[0]) < 0.70

train_x = X_train[train_test_split]

train_x = train_x[:6000]

train_y = Y_train[train_test_split]
test_x = X_train[~train_test_split]
test_y = Y_train[~train_test_split]


network = input_data(shape=[None, train_x.shape[1]])

network = fully_connected(network, 280, activation='tanh')
network = fully_connected(network, 300, activation='sigmoid')
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(train_x, train_y, n_epoch=2, shuffle=True, validation_set=(test_x, test_y),show_metric=True, batch_size=200, run_id='cifar10_cnn')
model.save("urban.tflearn")

model.load("urban.tflearn")
pred = model.predict(test_x)

print(pred)

if __name__ == "__main__":
    main()
