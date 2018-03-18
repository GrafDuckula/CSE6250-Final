import models_partc
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    kf = KFold(n=X.shape[0],n_folds=k)
    acc_list = []
    auc_list = []
    for train_idx, test_idx in kf:
        Y_pred = models_partc.logistic_regression_pred(X[train_idx],Y[train_idx],X[test_idx])
        acc, auc_, precision, recall, f1score = models_partc.classification_metrics(Y_pred,Y[test_idx])
        acc_list.append(acc)
        auc_list.append(auc_)
    acc_ave = mean(acc_list)
    auc_ave = mean(auc_list)
    return acc_ave,auc_ave


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    rs = ShuffleSplit(n=X.shape[0],n_iter=iterNo,test_size=test_percent)
    acc_list = []
    auc_list = []
    for train_idx, test_idx in rs:
        Y_pred = models_partc.logistic_regression_pred(X[train_idx],Y[train_idx],X[test_idx])
        acc, auc_, precision, recall, f1score = models_partc.classification_metrics(Y_pred,Y[test_idx])
        acc_list.append(acc)
        auc_list.append(auc_)
    acc_ave = mean(acc_list)
    auc_ave = mean(auc_list)
    return acc_ave,auc_ave


def main():

    # X,Y = utils.get_data_from_svmlight("../code_scala/output/withoutUPDRS.train")
    X,Y = utils.get_data_from_svmlight("../code_scala/output/withUPDRS.train")
    print "Classifier: Logistic Regression__________"
    acc_k,auc_k = get_acc_auc_kfold(X,Y)
    print "Average Accuracy in KFold CV: "+str(acc_k)
    print "Average AUC in KFold CV: "+str(auc_k)
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
    print "Average Accuracy in Randomised CV: "+str(acc_r)
    print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

