import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

from operator import itemgetter

import utils

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT
# USE THIS RANDOM STATE FOR ALL OF THE PREDICTIVE MODELS
# THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: X_train, Y_train and X_test
#output: Y_pred
def logistic_regression_pred(X_train, Y_train, X_test):
	#TODO: train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier	
    logreg = LogisticRegression() # create new class
    logreg.fit(X_train, Y_train) # train
    logregCoef = logreg.sparsify().coef_
    # print logreg.intercept_
    Z = logreg.predict(X_test) # predict
    # print sum(abs(Z - Y_train))
    return Z

#input: X_train, Y_train and X_test
#output: Y_pred
def svm_pred(X_train, Y_train, X_test):
	#TODO:train a SVM classifier using X_train and Y_train. Use this to predict labels of X_test
	#use default params for the classifier
    lsvc = LinearSVC() # create new class
    lsvc.fit(X_train, Y_train) # train
    Z = lsvc.predict(X_test) # predict
    # print sum(abs(Z - Y_train))
    return Z

#input: X_train, Y_train and X_test
#output: Y_pred
def decisionTree_pred(X_train, Y_train, X_test):
	#TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
	#IMPORTANT: use max_depth as 5. Else your test cases might fail.
    dTree = DecisionTreeClassifier(max_depth=5) # create new class
    dTree.fit(X_train, Y_train) # train
    Z = dTree.predict(X_test) # predict
    # print sum(abs(Z - Y_train))
    return Z


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#TODO: Calculate the above mentioned metrics
	#NOTE: It is important to provide the output in the same order
    accuracy = accuracy_score(Y_true, Y_pred)
    auc_ = roc_auc_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1score = f1_score(Y_true, Y_pred)
    return accuracy,auc_,precision,recall,f1score

#input: Name of classifier, predicted labels, actual labels
def display_metrics(classifierName,Y_pred,Y_true):
	print "______________________________________________"
	print "Classifier: "+classifierName
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print "Accuracy: "+str(acc)
	print "AUC: "+str(auc_)
	print "Precision: "+str(precision)
	print "Recall: "+str(recall)
	print "F1-score: "+str(f1score)
	print "______________________________________________"
	print ""

def main():
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../data/features_svmlight.validate")

	display_metrics("Logistic Regression",logistic_regression_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("SVM",svm_pred(X_train,Y_train,X_test),Y_test)
	display_metrics("Decision Tree",decisionTree_pred(X_train,Y_train,X_test),Y_test)

if __name__ == "__main__":
	main()
	
