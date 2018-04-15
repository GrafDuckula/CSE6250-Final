import models_partc
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477



def PC_analysis(X_train, X_test):

    svd = TruncatedSVD(n_components=35, n_iter=7, random_state=42)
    svd.fit(X_train)
    X_train_transformed = svd.transform(X_train)
    X_test_transformed = svd.transform(X_test)
    X_train_transformed_sparse = csr_matrix(X_train_transformed)
    X_test_transformed_sparse = csr_matrix(X_test_transformed)
    
    return X_train_transformed_sparse, X_test_transformed_sparse
    


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,algo="logistic regression",k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    kf = KFold(n=X.shape[0],n_folds=k)
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    
    for train_idx, test_idx in kf:
        # X_train, X_test = PC_analysis(X[train_idx], X[test_idx]) # PCA
        X_train, X_test = X[train_idx], X[test_idx] # without PCA
        if algo == "logistic regression":
            Y_pred = models_partc.logistic_regression_pred(X_train,Y[train_idx],X_test)
        elif algo == "linear_svm":
            Y_pred = models_partc.svm_pred(X_train,Y[train_idx],X_test)
        elif algo == "decision_tree":   
            Y_pred = models_partc.decisionTree_pred(X_train,Y[train_idx],X_test)
        elif algo == "ada boost":
            Y_pred = models_partc.ada_boost_pred(X_train,Y[train_idx],X_test)
        elif algo == "bagging logistic":
            Y_pred = models_partc.bagging_log_pred(X_train,Y[train_idx],X_test)
        elif algo == "bagging_svm":
            Y_pred = models_partc.bagging_SVC_pred(X_train,Y[train_idx],X_test)
        acc, auc_, precision, recall, f1score = models_partc.classification_metrics(Y_pred,Y[test_idx]) 
        acc_list.append(acc)
        auc_list.append(auc_)
        precision_list.append(precision)
        recall_list.append(recall)
    acc_ave = mean(acc_list)
    auc_ave = mean(auc_list)
    precision_ave = mean(precision_list)
    recall_ave = mean(recall_list)
    
    return acc_ave,auc_ave,precision_ave,recall_ave


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,algo="logistic regression", iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    rs = ShuffleSplit(n=X.shape[0],n_iter=iterNo,test_size=test_percent)
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []

    for train_idx, test_idx in rs:
        X_train, X_test = models_partc.PC_analysis(X[train_idx], X[test_idx]) # PCA
        # X_train, X_test = X[train_idx], X[test_idx] # without PCA
        if algo == "logistic regression":
            Y_pred = models_partc.logistic_regression_pred(X_train,Y[train_idx],X_test)
        elif algo == "linear_svm":
            Y_pred = models_partc.svm_pred(X_train,Y[train_idx],X_test)
        elif algo == "decision_tree":   
            Y_pred = models_partc.decisionTree_pred(X_train,Y[train_idx],X_test)
        elif algo == "ada boost":
            Y_pred = models_partc.ada_boost_pred(X_train,Y[train_idx],X_test)
        elif algo == "bagging logistic":
            Y_pred = models_partc.bagging_log_pred(X_train,Y[train_idx],X_test)
        elif algo == "bagging_svm":
            Y_pred = models_partc.bagging_SVC_pred(X_train,Y[train_idx],X_test)
        acc, auc_, precision, recall, f1score = models_partc.classification_metrics(Y_pred,Y[test_idx])
        acc_list.append(acc)
        auc_list.append(auc_)
        precision_list.append(precision)
        recall_list.append(recall)
    acc_ave = mean(acc_list)
    auc_ave = mean(auc_list)
    precision_ave = mean(precision_list)
    recall_ave = mean(recall_list)
    return acc_ave,auc_ave,precision_ave,recall_ave


def main():
    
    for algo in ["logistic regression", "linear_svm", "decision_tree", "ada boost", "bagging logistic", "bagging_svm"]:
        X,Y = utils.get_data_from_svmlight("../scala/output/withoutUPDRS.train")
        print 
        print "Without UPDRS"
        print "Classifier:", algo, "__________"
        acc_k,auc_k, precision_k,recall_k = get_acc_auc_kfold(X,Y,algo)
        print "Average Accuracy in KFold CV: "+str(acc_k)
        print "Average AUC in KFold CV: "+str(auc_k)
        print "Average Precision Score in KFold CV: "+str(precision_k)
        print "Average Recall Score in KFold CV: "+str(recall_k)
 
        X,Y = utils.get_data_from_svmlight("../scala/output/withUPDRS.train")
        print "With UPDRS"
        print "Classifier:", algo, "__________"
        acc_k,auc_k, precision_k,recall_k = get_acc_auc_kfold(X,Y,algo)
        print "Average Accuracy in KFold CV: "+str(acc_k)
        print "Average AUC in KFold CV: "+str(auc_k)
        print "Average Precision Score in KFold CV: "+str(precision_k)
        print "Average Recall Score in KFold CV: "+str(recall_k)
        
        
#    X,Y = utils.get_data_from_svmlight("../scala/output/withoutUPDRS.train")
#    print "Without UPDRS"
#    print "Classifier: Logistic Regression__________"
#    acc_k,auc_k = get_acc_auc_kfold(X,Y)
#    print "Average Accuracy in KFold CV: "+str(acc_k)
#    print "Average AUC in KFold CV: "+str(auc_k)
#    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
#    print "Average Accuracy in Randomised CV: "+str(acc_r)
#    print "Average AUC in Randomised CV: "+str(auc_r)
#    
#    X,Y = utils.get_data_from_svmlight("../scala/output/withUPDRS.train")
#    print "With UPDRS"
#    print "Classifier: Logistic Regression__________"
#    acc_k,auc_k = get_acc_auc_kfold(X,Y)
#    print "Average Accuracy in KFold CV: "+str(acc_k)
#    print "Average AUC in KFold CV: "+str(auc_k)
#    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
#    print "Average Accuracy in Randomised CV: "+str(acc_r)
#    print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

