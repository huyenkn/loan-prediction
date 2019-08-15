# loan-prediction
This repository is part of a loan eligibility process based on customer details provided after customers complete their online loan application form. My target is to predict the customers in the dataset who are eligible for loan approval. 
Evaluation Metric is accuracy (percentage of the loan approvals that are correctly predicted)

The training and test set are Pandas DataFrames (read from CSV files, whose shapes are (614, 13) and (367, 12) respectively). Each row contains features for a customer. Each column is a feature (Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others...).
LoanAmount, ApplicantIncome, and CoapplicantIncome columns are continuous; the others are categorical. 

At best, the model achieves ...% accuracy.

Before training the data, I implemented munging data process:
	1. Detecting missing values
	Detect and fix different types of missing values.
	
	2. Filling missing values
	2.1. Fill missing values in categorical columns.
	Use 'Mode' to fill missing values in categorical columns because:
	- Mode is suitable for categorical data
	- The number of missing values in each column is very small compared to the total population (below 5%)
	- The proportion of each answer's type is large
	- Every variable presumably does not have any interrelationships with others and is treated individually.

	The approach to handle missing values heavily depend on the nature of the dataset, so if the data is large and do not meet the above criteria, you could use advanced predictive techniques (machine learning algorithms). In that case, missing values are filled by exploring their correlations among others.

	2.2. Find the median value of continuous columns by group.
	Use Median by group because:
	- Median is suitable for continuous data
	- The number of missing values in each column is very small compared to the total population (below 5%)
	- By investigating some dependency of a variable with others, this method allows better estimation than using global median value.

	3. Construct new features
	Use log-transformation to address skewed data/extreme values/outliers.

	4. Convert categorical features to numeric. 
	Convert all categorical variables into numeric since sklearn requires all inputs to be numeric.

