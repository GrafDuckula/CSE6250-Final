import models_partc
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS
# VALIDATION TESTS OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477



def PC_analysis(X_train, X_test, n_components=200):

    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=RANDOM_STATE)
    svd.fit(X_train)
    X_train_transformed = svd.transform(X_train)
    X_test_transformed = svd.transform(X_test)
    X_train_transformed_sparse = csr_matrix(X_train_transformed)
    X_test_transformed_sparse = csr_matrix(X_test_transformed)
    
    return X_train_transformed_sparse, X_test_transformed_sparse
    


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,algo="logistic regression",k=5, n_components=110):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    kf = KFold(n=X.shape[0],n_folds=k)
    acc_list = []
    auc_list = []
    precision_list = []
    recall_list = []
    
    for train_idx, test_idx in kf:
        # X_train, X_test = PC_analysis(X[train_idx], X[test_idx], n_components) # PCA
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
        elif algo == "neural_network":
            Y_pred = models_partc.neural_network(X_train, Y[train_idx], X_test)
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
#def get_acc_auc_randomisedCV(X,Y,algo="logistic regression", iterNo=5,test_percent=0.2):
#	#TODO: First get the train indices and test indices for each iteration
#	#Then train the classifier accordingly
#	#Report the mean accuracy and mean auc of all the iterations
#    
#    rs = ShuffleSplit(n=X.shape[0],n_iter=iterNo,test_size=test_percent)
#    acc_list = []
#    auc_list = []
#    precision_list = []
#    recall_list = []
#
#    for train_idx, test_idx in rs:
#        X_train, X_test = models_partc.PC_analysis(X[train_idx], X[test_idx]) # PCA
#        # X_train, X_test = X[train_idx], X[test_idx] # without PCA
#        if algo == "logistic regression":
#            Y_pred = models_partc.logistic_regression_pred(X_train,Y[train_idx],X_test)
#        elif algo == "linear_svm":
#            Y_pred = models_partc.svm_pred(X_train,Y[train_idx],X_test)
#        elif algo == "decision_tree":   
#            Y_pred = models_partc.decisionTree_pred(X_train,Y[train_idx],X_test)
#        elif algo == "ada boost":
#            Y_pred = models_partc.ada_boost_pred(X_train,Y[train_idx],X_test)
#        elif algo == "bagging logistic":
#            Y_pred = models_partc.bagging_log_pred(X_train,Y[train_idx],X_test)
#        elif algo == "bagging_svm":
#            Y_pred = models_partc.bagging_SVC_pred(X_train,Y[train_idx],X_test)
#        acc, auc_, precision, recall, f1score = models_partc.classification_metrics(Y_pred,Y[test_idx])
#        acc_list.append(acc)
#        auc_list.append(auc_)
#        precision_list.append(precision)
#        recall_list.append(recall)
#    acc_ave = mean(acc_list)
#    auc_ave = mean(auc_list)
#    precision_ave = mean(precision_list)
#    recall_ave = mean(recall_list)
#    return acc_ave,auc_ave,precision_ave,recall_ave


def main():
    
   #for algo in ["logistic regression", "linear_svm", "decision_tree", "ada boost", "bagging logistic", "bagging_svm", "neural_network"]:
   for algo in ["neural_network"]:
       X,Y = utils.get_data_from_svmlight("../scala/output/withoutUPDRS.train", n_features=400)
       print 
       print "Without UPDRS"
       print "Classifier:", algo, "__________"
       acc_k,auc_k, precision_k,recall_k = get_acc_auc_kfold(X,Y,algo)
       print acc_k
       print auc_k
       print precision_k
       print recall_k

#        print "Average Accuracy in KFold CV: "+str(acc_k)
#        print "Average AUC in KFold CV: "+str(auc_k)
#        print "Average Precision Score in KFold CV: "+str(precision_k)
#        print "Average Recall Score in KFold CV: "+str(recall_k)

       X,Y = utils.get_data_from_svmlight("../scala/output/withUPDRS.train", n_features=400)
       print "With UPDRS"
       print "Classifier:", algo, "__________"
       acc_k,auc_k, precision_k,recall_k = get_acc_auc_kfold(X,Y,algo)
       print acc_k
       print auc_k
       print precision_k
       print recall_k
        
       # print "Average Accuracy in KFold CV: "+str(acc_k)
       # print "Average AUC in KFold CV: "+str(auc_k)
       # print "Average Precision Score in KFold CV: "+str(precision_k)
       # print "Average Recall Score in KFold CV: "+str(recall_k)
  
    # for algo in ["logistic regression", "linear_svm", "decision_tree"]:
    #     without_updrs = []
    #     with_updrs = []
    #     # n_components = np.arange(10, 340, 10)
    #     n_components = np.arange(5, 140, 5)
    #     for n in n_components:
    #         X,Y = utils.get_data_from_svmlight("../scala/output/withoutUPDRS.train", n_features=400)
    #         acc_k,auc_k, precision_k,recall_k = get_acc_auc_kfold(X,Y,algo,k=5, n_components=n)
    #         without_updrs.append(acc_k)
            
    #         X,Y = utils.get_data_from_svmlight("../scala/output/withUPDRS.train", n_features=400)
    #         acc_k,auc_k, precision_k,recall_k = get_acc_auc_kfold(X,Y,algo,k=5, n_components=n)
    #         with_updrs.append(acc_k)
        
    #     print "Classifier:", algo, "__________"
    #     # print "Without UPDRS"
    #     print np.argmax(without_updrs), np.max(without_updrs)
    #     # print "With UPDRS"
    #     print np.argmax(with_updrs), np.max(with_updrs)
    #     plt.plot(n_components, without_updrs, label=algo+' without_updrs')
    #     plt.plot(n_components, with_updrs, label=algo+' with_updrs')
    # plt.xlabel('nb of components')
    # plt.ylabel('ACC')
    # plt.legend(loc='lower right')  
    # plt.show()
        
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

