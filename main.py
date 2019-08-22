import os
import sys
import random
import argparse
import time
import random 
random.seed(1234)

import numpy as np
np.random.seed(1234)
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import flags


def preprocess_data(data):
    """
        Munge raw input data.
        Input:
            - data: a Pandas DataFrame (read from a CSV file). 
        Output: None
    """
    def pp_continuous_amount(x, table):
        """
        Select a value from the pivot table based on a data example's values of the
        Self_Employed and Education features.
        Input:
            x: a data example (a row of the input DataFrame)
        Output: a value selected from the pivot table (numpy.float64)  
        """ 
        return table.loc[x['Self_Employed'], x['Education']]    
    
    # 1. Detecting missing values   
    string_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for column in string_columns:
        for i, value in enumerate(data[column]):
            if not str(value).replace(' ', '').isalpha():
                data[i, column] = np.nan

    # 2. Fill missing values
    # 2.1. Fill missing values in categorical columns.
    continuous_sets = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
    for column in data.columns:
        if column not in continuous_sets and data[column].isnull().sum() > 0:
            data[column].fillna(data[column].mode()[0], inplace=True)

    # 2.2. Find the median value of continuous columns by group.
    table = {}
    for column in continuous_sets:
        table[column] = data.pivot_table(values=column, index='Self_Employed', 
            columns='Education', aggfunc=np.median)

    for column in continuous_sets:
        continuous_amount_nulls = data[column].isnull()
        if continuous_amount_nulls.sum() > 0:
            data[column].fillna(
                data[continuous_amount_nulls].apply(
                    lambda x: pp_continuous_amount(x, table[column]), axis=1), inplace=True)    

    # 3. Construct new features
    data['LoanAmount_log'] = np.log(data.LoanAmount)
    data['TotalIncome'] = data.ApplicantIncome + data.CoapplicantIncome
    data['TotalIncome_log'] = np.log(data.TotalIncome)

    # 4. Convert categorical features to numeric. 
    var_mod = ['Gender','Married','Dependents','Education',
               'Self_Employed','Property_Area', 'LoanAmount_log']
    for i in var_mod:
        data[i] = LabelEncoder().fit_transform(data[i])

def train (data, predictors, outcome, method):
    """
        Take one of below models as input and calculate the Accuracy and Cross-Validation scores 
        of the training set.
        input: 
        - data: the training set (a dataframe in csv file)
        - predictors: attributes of the dataframe used to predict the outcome.
        - outcome: what we want to predict (loan status).
        - method: random forest (default), decision tree, or logistic regression.
    """ 
    if method == 'decision_tree':
        train_model = DecisionTreeClassifier()
 
    elif method == 'log_reg':
        train_model = LogisticRegression(solver='lbfgs')
    
    else:
        train_model = RandomForestClassifier(n_estimators=25, min_samples_split=25, 
        max_depth=7, max_features=1)

    # Apply preprocess_data() function to process data.
    preprocess_data(data)

    # Fit the model:
    train_model.fit(data[predictors], data[outcome])

    # Make predictions on the training set:
    predictions = train_model.predict(data[predictors])

    # Use accuracy metric to evaluate the model on the training set:
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))  
    

    #Perform k-fold cross-validation with 5 folds:
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    error = []

    for train, test in kf.split(data[predictors]): 
        # Filter the training set:
        train_predictors = (data[predictors].iloc[train, :])

        # Filter the target that we are using to train the algorithm:
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target:
        train_model.fit(train_predictors, train_target)

        # Record error from each cross-validation run:
        error.append(train_model.score(
            data[predictors].iloc[test,:], data[outcome].iloc[test]))

    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be refered outside the function:
    train_model.fit(data[predictors],data[outcome])
    # Create a series with feature importances:
    if method != "log_reg":
        featimp = pd.Series(train_model.feature_importances_, 
            index=predictor_var).sort_values(ascending=False)
    print ('Feature importances: \n', featimp, sep="")
    return train_model

def test(data, predictors, model):

    preprocess_data(data)
    return model.predict(data[predictors])

if __name__ == "__main__":

    parser = flags.make_parser()
    args = parser.parse_args()

    # Find out whether the training and test files exist:
    if not os.path.exists(args.train_file):
        print('Train file %s not exists!!!' % args.train_file)
        sys.exit(1)

    if not os.path.exists(args.test_file):
        print('Test file %s not exists!!!' % args.test_file)
        sys.exit(1)

    # Input the training and test files:
    missing_values = ["n/a", "na", "--", "-", "N/A"]
    df_train = pd.read_csv(args.train_file, na_values = missing_values)
    df_test = pd.read_csv(args.test_file, na_values = missing_values)

    #Create a series with feature importances:
    
    outcome_var = 'Loan_Status'
    predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
    train_model = train(df_train, predictor_var, outcome_var, args.method)
    predictions = test(df_test, predictor_var, train_model)

    # Save and format the result:
    result = pd.DataFrame({'Loan_ID' : df_test.Loan_ID, 
    'Loan_Status' : ['Y' if x == 1 else 'N' for x in predictions]})

    result.to_csv()
    with open('test_predictions.csv', 'w') as f:
        print(result.to_csv(index=False), file=f)




