# CSE-6250-Final
Parkinson's Disease
Requiments: (install specific versions when specified)
1) install spark and scala 
2) Python 2.7 : sklearn, scipy, numpy, matplotlib
3) Python 3.0 : tensorflow, keras 2.1.0

## Running Spark Preprocessing
`sbt compile run`

This will output all training/test files to `scala/output` directory

## Running logistic regression/svm/decision tree/ada boost/bagging logistic regression/bagging svc/neural network
`python cross.py`

Output:
Terminal console will output evaluation results (5-fold cross validation)

## Enabling PCA
* **Without PCA** Comment out `line 48` and use `line 47` of `cross.py`
* **With PCA** Comment out `line 47` and use `line 48` of `cross.py`

## Running LSTM model
Edit filepath for lstm training/testing data with variable: `LSTM_DATA_PATH`
`python3 nn_lstm.py`

Output:
Terminal console will output evaluation results (5-fold cross validation) averaged in the order of:
`[accuracy, auc, sensitivity, specificity]`


