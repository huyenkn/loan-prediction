# loan-prediction
 
This repository is based on the [Analytics Vidhya tutorial](https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/) on its [loan prediction problem](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/). I implement scripts to (a) download the loan prediction data and (b) run several machine learning methods (random forest, logistic regression, decision tree) on this problem. 

The loan prediction problem predicts loan eligibility based on customer information provided after customers complete their online loan application forms. Performance is measured by accuracy (percentage of the loan approvals that are correctly predicted). I apply three methods (from `sklearn`): Logistic regression, Random forest, and Decision tree. The best method, logistic regression, achieves **80.9% in accuracy** with 5-fold cross-validation. 

### Data format

The training and test set are Pandas DataFrames (read from CSV files, whose shapes are (614, 13) and (367, 12) respectively). Each row contains features for a customer. Each column is a feature (gender, marital status, education, number of dependents, income, loan amount, credit history, ...).
LoanAmount, ApplicantIncome, and CoapplicantIncome columns are continuous; the others are categorical. 

### Data pre-processing

Data is pre-processed as follows:

1. Detect and standardize different types of missing values.
	
2. Fill in missing values

2.1. Fill in missing values of categorical columns.

Replace missing values with mode (most frequent) values of the corresponding columns. Using 'mode' because:
- Mode is suitable for categorical data;
- The number of missing values in each column is small compared to the total population (below 5%);
- Data is very imbalanced;

In using 'mode', I assume that the imputed variables do not have any correlation with the measured variables. 
The approach to handling missing values heavily depends on the nature of the dataset.
Hence, if the data seriously violate this assumption, one may consider using more advanced predictive techniques (e.g. machine learning algorithms) to exploit the correlations among variables.

2.2. Fill in missing values of continuous columns.

Replace missing values with grouped median values. For example, with the "LoanAmount" column, I compute the median value for every pair of values in the "Self_Employed" and "Education" columns. Then each missing value is replaced by the median value corresponding to the values of the measured "Self_Employed" and "Education" values in the same row.  Use `grouped median` because:
- Median is suitable for continuous data
- The number of missing values in each column is small compared to the total population (below 5%)
- Taking into account dependency of a variable on others results in better estimation than treating it as being independent from others.

3. Construct new features 
Use log-transformation to reduce skewness in columns "LoanAmount" and "TotalIncome".

4. Convert categorical features to numeric. 
Convert all categorical variables into numeric because `sklearn` requires all inputs to be numeric.

