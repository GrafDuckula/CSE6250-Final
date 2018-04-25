
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
import glob

def main():
    print("Using NN to train")
    (X_train, Y_train) = read_train_data()
    Y_train = one_hot_encode(np.asarray(Y_train))
    train_test_split = np.random.rand(X_train.shape[0]) < 0.90
    X_train_split = X_train[train_test_split]
    Y_train_split = Y_train[train_test_split]
    X_test_split = X_train[~train_test_split]
    Y_test_split = Y_train[~train_test_split]
    train_model(X_train_split, Y_train_split, X_test_split, Y_test_split)

	# train_test_split = np.random.rand(X_train.shape[0]) < 0.70
	# train_x = X_train[train_test_split]
	# train_x = train_x[:6000]
	# pred = model.predict(test_x)
	# train_y = Y_train[train_test_split]
	# test_x = X_train[~train_test_split]
	# test_y = Y_train[~train_test_split]
	# print(pred)


def one_hot_encode(labels):
	print(labels[0])
	n_labels = len(labels)
	print(n_labels)
	n_unique_labels = len(np.unique(labels))
	print(n_unique_labels)
	one_hot_encode = np.zeros((n_labels,n_unique_labels))
	print(one_hot_encode.shape)
	print(one_hot_encode)
	for i in range(0,len(labels)):
		if labels[i] == 0.0:
			one_hot_encode[i, 0] = 1.0
			one_hot_encode[i, 1] = 0.0
		else:
			one_hot_encode[i, 0] = 0.0
			one_hot_encode[i, 1] = 1.0
	return one_hot_encode
    

def read_train_data():
	X_train = []
	file_list = glob.glob('../scala/output/X_train1.csv' + '/*')
	for file_path in file_list:
		if 'part' in file_path:
			temp = np.genfromtxt(file_path, delimiter=',')
			if len(X_train) is 0:
				X_train = temp
			else:
				X_train = np.concatenate((X_train, temp))

	Y_train = []
	file_list = glob.glob('../scala/output/Y_train1.csv' + '/*')
	for file_path in file_list:
		if 'part' in file_path:
			temp = np.genfromtxt(file_path, delimiter=',')
			if len(Y_train) is 0:
				Y_train = temp
			else:
				Y_train = np.concatenate((Y_train, temp))
	print(X_train.shape, Y_train.shape)
	return (X_train,Y_train)

def train_model(X_train, Y_train, X_test, Y_test):
	print(X_train.shape)
	print(Y_train.shape)
	print(X_test.shape)
	print(Y_test.shape)
	tflearn.init_graph(num_cores=8)
	network = input_data(shape=[None, X_train.shape[1]])
	network = fully_connected(network, 280, activation='sigmoid')
	network = fully_connected(network, 300, activation='sigmoid')
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.1)
	model = tflearn.DNN(network, tensorboard_dir='tmp/dir', tensorboard_verbose=0)
	model.fit(X_train, Y_train, n_epoch=55, shuffle=True, validation_set=(X_test, Y_test),show_metric=True, batch_size=50, run_id='ppmi')
	model.save("ppmi.tflearn")
	model.load("ppmi.tflearn")

# def test_model():

	




if __name__ == "__main__":
    main()
