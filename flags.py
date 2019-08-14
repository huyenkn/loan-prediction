import argparse

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', type=str, required=True,
    	help="Path to the training file (CSV only)")
    parser.add_argument('-test_file', type=str, required=True, 
    	help="Path to the test file (CSV only)")
    parser.add_argument('-method', type=str, choices=['log_reg', 'random_forest', 'decision_tree'], default='random_forest', 
    	help="choose one method for training: log_reg (logistic regression), random_forest (random forest), decision_tree (decision tree)")
    parser.add_argument('-output', type=str, required=True,
    	help="Path for saving the predictions on the test_file (CSV format)")

    return parser