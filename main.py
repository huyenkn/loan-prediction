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
		Munge raw input data:
		1. Detecting missing values
		2. Filling missing values
		3. Construct new features
		4. Convert categorical features to numeric. 

		Input:
			- data: a Pandas DataFrame (read from a CSV file). Each row contains features for a customer. 
			Each column is a feature (Gender, Marital Status, Education, Number of Dependents, Income, 
			Loan Amount, Credit History and others...). 
			LoanAmount, ApplicantIncome, and CoapplicantIncome are continuous; others are categorical. 

		Output: None
	
	1. Filling in missing values
	- For discrete data: the proportion of answer's types are very large, so we can impute the missing
	  value with median, mean or mode values of the column while can relatively maintain 
	  the initial ratios. (in this case, using mean/median still gets the same values as using mode)

	- For continuous data: we assume that there is correlations between attributes with missing values 
	  and other attributes in the dataset, then basing on these correlations we estimate the missing value.
	  We're going to use Self_Employed and Education to estimate the missing values of LoanAmount (we could
	  do the same things with different combinations of other attributes ApplicantIncome, CoapplicantIncome).
	"""

	# 1. Detecting missing values
	# Detect and fix different types of missing values:
	string_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
	for column in string_columns:
		for i, value in enumerate(data[column]):
			if not str(value).replace(' ', '').isalpha():
				data[i, column] = np.nan

	def pp_continuous_amount(x, table):

		"""
		Select a value from the pivot table based on a data example's values of the
		Self_Employed and Education features.

		Input:
			x: a data example (a row of the input DataFrame)
		Output: a value selected from the pivot table (numpy.float64)  
		""" 
		return table.loc[x['Self_Employed'], x['Education']]
		
	# 2. Fill missing values in categorical columns. 
	"""
	Use 'Mode' to fill missing values in categorical columns because:
	- The number of missing values in each column is very small compared to the total population 
	  (all columns are below 5%)
	- The proportion of each answer's type is large
	- Mode is suitable for categorical data
	- Every variable is treated individually, and presumably does not have any interrelationships with others.
	
	The approach to handle missing values heavily depend on the nature of the dataset, 
	so in case the data is large and do not meet the above standards, you can use advanced predictive techniques
	(machine learning algorithms) in which missing values are filled by exploring their correlations.
	"""

	continuous_sets = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']
	for column in data.columns:
		if column not in continuous_sets and data[column].isnull().sum() > 0:
			data[column].fillna(data[column].mode()[0], inplace=True)

	# 3. Find the median value of continuous columns for each value pair (Self_Employed, Education).
	"""
	Use Median by groups because:
	- The number of missing values in each column is very small compared to the total population 
	  (all columns are below 5%)
	- Median is suitable for continuous data
	- 

	"""

	table = {}
	for column in continuous_sets:
		table[column] = data.pivot_table(values=column, index='Self_Employed', 
			columns='Education', aggfunc=np.median)

	"""
	To match the values from 'table' to all cells coordinated by Self_Employed and Education.
	In this case, this function is applied to the attribute Dataframe.apply()
	Input: Dataframe (the sample set)
	Output: Dataframe with its dimension n * n (n is the total number of rows of a sample set).
	"""
	
	# 1. Fill missing values in the LoadAmount column. 

	for column in continuous_sets:
		continuous_amount_nulls = data[column].isnull()
		if continuous_amount_nulls.sum() > 0:
			data[column].fillna(
				data[continuous_amount_nulls].apply(
					lambda x: pp_continuous_amount(x, table[column]), axis=1), inplace=True)	

	# 2. Using log-transformation to address skewed data/extreme values/outliers 

	data['LoanAmount_log'] = np.log(data.LoanAmount)
	data['TotalIncome'] = data.ApplicantIncome + data.CoapplicantIncome
	data['TotalIncome_log'] = np.log(data.TotalIncome)

	#print(data)


	# 3. Converting all categorical variables into numeric.
	var_mod = ['Gender','Married','Dependents','Education',
	           'Self_Employed','Property_Area', 'LoanAmount_log']
	for i in var_mod:
		data[i] = LabelEncoder().fit_transform(data[i])

	"""
	takes one of three models below as input and determines the Accuracy and Cross-Validation scores
	of the training set.
	input: 
	data: the training set (a dataframe in csv file)
	predictors: attributes of the dataframe used to predict the outcome.
	outcome: what we want to predict (loan status).
	method: random forest (default), decision tree, or logistic regression.
	"""

def train (data, predictors, outcome, method):

    
    if method == 'decision_tree':
    	train_model = DecisionTreeClassifier()
 
    elif method == 'log_reg':
    	train_model = LogisticRegression(solver='lbfgs')
    
    else:
    	train_model = RandomForestClassifier(n_estimators=25, min_samples_split=25, 
    	max_depth=7, max_features=1)
    #print(method, train_model)

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

	# 
	outcome_var = 'Loan_Status'
	predictor_var = ['TotalIncome_log', 'LoanAmount_log', 'Credit_History']
	train_model = train(df_train, predictor_var, outcome_var, args.method)
	predictions = test(df_test, predictor_var, train_model)

	# Save and format the result:
	result = pd.DataFrame({'Loan_ID' : df_test.Loan_ID, 
	'Loan_Status' : ['Y' if x == 1 else 'N' for x in predictions]})

	result.to_csv()
	with open('test_predictions.csv', 'w') as f:
		print(result.to_csv(index=False), file=f)




