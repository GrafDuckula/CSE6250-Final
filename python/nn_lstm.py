import pandas as pd
import numpy as np
import keras
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from models_partc import classification_metrics

NUM_TIME_STAMP = 16
NUM_FEATURES = 346
LSTM_DATA_PATH = '../scala/output/withUPDRS_LSTM.train'

def shuffle_data(X, Y):
    rows = Y.shape[0]
    print(rows)
    indices = np.asarray([i for i in range(rows)])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    return X, Y

def k_fold_split_indices(X, k):
    splits = []
    rows = X.shape[0]

    per_iter = math.floor(rows/k)
    start_index = 0
    end_index = per_iter

    for i in range(k):
        split = [i for i in range(start_index, end_index)]
        splits.append(split)
        start_index = end_index + 1
        if i < k - 1:
            end_index = end_index + per_iter + 1
        else:
            end_index = rows

    return splits

def getTrainTest(split, X, Y):
    split = np.asarray(split)
    X_test = X[split]
    X_train = X[~split]
    Y_test = Y[split]
    Y_train = Y[~split]

    return X_train, X_test, Y_train, Y_test

def getAverage(scores):
    sum = [0 for i in range(len(scores[0]))]
    for score in scores:
        for i, metric in enumerate(score):
            sum[i] += metric
    
    for i, metric in enumerate(sum):
        sum[i] = sum[i] / len(scores)

    return sum

def get_data(path):
    f = open(path, 'r')
    return get_impute_data(f)

def get_impute_data(f):
    labels = []
    data = []
    for line in f:
        details = line.split(" ")
        labels.append(int(details[0]))

        ans = np.zeros((NUM_FEATURES, NUM_TIME_STAMP))
        for i in range(1, len(details)):
            detail = details[i]
            detail = detail.split(",")
            timestamp, feature, value = detail
            ans[int(feature)][int(timestamp)] = float(value)
        data.append(ans)
    return np.asarray(data), np.asarray(labels)

def get_pad_data(f):
    labels = []
    data = []
    for line in f:
        details = line.split(" ")
        labels.append(int(details[0]))

        ans = np.zeros((NUM_FEATURES, NUM_TIME_STAMP))
        
        map_pad_index = {}
        next_highest_index = 0

        for i in range(1, len(details)):
            detail = details[i]
            detail = detail.split(",")
            timestamp, feature, value = detail


            if timestamp in map_pad_index:
                timestamp = map_pad_index[timestamp]
            else:
                map_pad_index[timestamp] = next_highest_index
                next_highest_index += 1
                timestamp = map_pad_index[timestamp]

            ans[int(feature)][timestamp] = float(value)
        data.append(ans)
    return np.asarray(data), np.asarray(labels)


def train(X_train, Y_train, X_test):
    model = Sequential()
    model.add(Dense(32, input_shape=(NUM_FEATURES,NUM_TIME_STAMP)))
    # model = Sequential()
    model.add(LSTM(200))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, Y_train, nb_epoch=3, batch_size=9)
    Y_pred = model.predict_classes(X_test)
    row = Y_pred.shape[0]
    print(Y_pred.shape)
    Y_pred = Y_pred.reshape(row)
    print(Y_pred.shape)
    return Y_pred

def main():
    X, Y = get_data(LSTM_DATA_PATH)
    X, Y = shuffle_data(X, Y)
    splits = k_fold_split_indices(X, 5)
    all_scores = []
    for i in range(5):
        split = splits[i]
        X_train, X_test, Y_train, Y_test = getTrainTest(split, X, Y)
        Y_pred = train(X_train, Y_train, X_test)
        acc, auc_, precision, recall, f1score = classification_metrics(Y_pred, Y_test)
        all_scores.append([acc, auc_, precision, recall, f1score])
    print('-------FINAL-------')
    print(getAverage(all_scores))

if __name__ == "__main__":
	main()
	