# loan-prediction
 
This repository is based on the [Analytics Vidhya tutorial](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) on its [loan prediction problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/). I implement scripts to (a) download the loan prediction data and (b) run several machine learning methods (random forest, logistic regression, decision tree) on this problem. 

The loan prediction problem predicts loan eligibility based on customer information provided after customers complete their online loan application forms. Performance is measured by accuracy (percentage of the loan approvals that are correctly predicted).

The training and test set are Pandas DataFrames (read from CSV files, whose shapes are (614, 13) and (367, 12) respectively). Each row contains features for a customer. Each column is a feature (Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others...).
LoanAmount, ApplicantIncome, and CoapplicantIncome columns are continuous; the others are categorical. 

In this repository, I applied three model algorithms: Logistic regression, Random forest, and Decision tree.
At best, the model achieves 80,946% accuracy.

Before training the data, I implemented the following munging data process:
1. Detecting missing values
Detect and fix different types of missing values.
	
2. Filling missing values

2.1. Fill missing values in categorical columns.
Use 'Mode' to fill missing values in categorical columns since:
- Mode is suitable for categorical data
- The number of missing values in each column is very small compared to the total population (below 5%)
- The proportion of each answer's type is large
- Every variable presumably does not have any interrelationships with others and is treated individually.

The approach to handle missing values heavily depends on the nature of the dataset, so if the data are large and do not meet the above criteria, you could use more advanced predictive techniques (e.g. machine learning algorithms). In that case, missing values are filled by exploring the correlation of their variables among others.

2.2. Fill missing values in continuous columns.
Use Median by group since:
- Median is suitable for continuous data
- The number of missing values in each column is very small compared to the total population (below 5%)
- By investigating some dependency of a variable on others, this method allows better estimation than using only global median value.

3. Construct new features
Use log-transformation to address skewed data/extreme values/outliers.

4. Convert categorical features to numeric. 
Convert all categorical variables into numeric since sklearn library requires all inputs to be numeric.

