import numpy as np 
import pandas as pd
import random
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


def sign(x):
    return 2 * (x >= 0) - 1

class SimpleRandomForest:
    """Simple RandomForest class"""
    
    def __init__(self, num_estimators, max_depth=None, max_features=None):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int. The number of decision trees
                max_depth    : int. 
                classifiers  : list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[0]) is the dictionary denoted as classifier.
                             The classifier has key 'feature_indices' and 'classifier'. The values are the indices of features 
                             for that classifier and the Decsion Tree classifier.
                             
        Args:
            
        Returns:
        
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.max_depth = max_depth
        self.classifiers = []
        
        
    def fit(self, X, y):
        """
        Description:
            Build classifiers from the training set.
               
        Args:
            X: (N, d) numpy array. Training/testing samples.
            y: (N,) numpy array. Ground truth classes.
            
        Returns:
            
            
        """
        self.X = X
        self.y = y

        self.bagging()
        
        
    def bagging(self):
        """
        Description:
            Build simple randomforest classifier. Follow the procedures described in .ipynb file.
            Bagging - Each tree is grown using a bootstrap sample of training data

            Use DecisionTreeClassifier(max_depth=self.max_depth) as a decision tree.
            # step 1. Create random vectors
            # step 2. Use random vector to build multiple decision trees
            # step 3. Combine decision trees
        
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###
        for _ in range(self.num_estimator):
            # Get bootstrap samples and select random features
            X_bootstrapped, y_bootstrapped = self.sample_random_vectors()
            random_indices = self.sample_random_features()
            # Create new dataset using only selected features
            X_bootstrappped_feature = X_bootstrapped[:, random_indices]
            
            # Create and train decision tree
            decision_tree_i = DecisionTreeClassifier(max_depth=self.max_depth)
            decision_tree_i.fit(X_bootstrappped_feature, y_bootstrapped)
            
            # Store trained tree and its feature indices
            classifier_info = {'feature_indices': (np.array)(random_indices), 'classifier': decision_tree_i}
            self.classifiers.append(classifier_info)
        #################
        
    
    def sample_random_vectors(self):
        """
        Description:
            Select N random vectors with replacement for learning each decision tree.
            You may need to use 'np.random' or 'random' library to select arbitrary vectors.
            Note that N is row numbers of self.X.
        
        Args:
            
        Returns:
            X_bootstrapped: (N, d) numpy array.
            y_bootstrapped: (N,) numpy array.
            
        """
        ### CODE HERE ###
        # Get data size
        n_samples = self.X.shape[0]
        # Randomly select samples with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        # Create bootstrap samples using selected indices
        X_bootstrapped = self.X[indices]
        y_bootstrapped = self.y[indices]
        #################
        return X_bootstrapped, y_bootstrapped
    
    
    def sample_random_features(self):
        """
        Description:
            Select random features for learning each decision tree.
            You may need to use 'np.random' or 'random' library to select arbitrary features.
            Make sure that m is a random integer value, and selected m features are also random. 
            Note that d is column numbers of self.X.
            
            !! NOTICE that seleting random features DO NOT allow replacement !!
        
        Args:
            
        Returns:
            random_incides: (m,) numpy array. m (<=d and >=1) is an integer value.
            
            
        """
        ### CODE HERE ###
        # Get total number of features
        d = self.X.shape[1]
        # Select random number of features between 1 and d
        m = np.random.randint(1, d+1)
        # Select m random feature indices without replacement
        random_indices = np.random.choice(d, size=m, replace=False)
        #################
        return random_indices
        
        
    def predict(self, X):
        """
        Description:
            Predict the target variables (RandomForest's final prediction). Use the results of voting among attribute classifiers. (i.e. use voting method)
            
            The dictionary {key: value} is composed,
                classifier : {'feature_indices': (indices),
                              'classifier' : (DecisionTreeClassifier)}
               
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of RandomForest classifier. Output values are of 1 or -1.
            
        """
        
        ### CODE HERE ###
        # Initialize array for predictions
        pred = np.zeros(X.shape[0])
        for classifier_info in self.classifiers:
            # Get feature indices and classifier for each tree
            feature_indices = classifier_info['feature_indices']
            classifier = classifier_info['classifier']
            # Make predictions using only selected features
            X_features = X[:, feature_indices]
            prediction_i = classifier.predict(X_features)
            # Accumulate predictions
            pred += prediction_i
        
        # Determine final prediction (majority voting)
        pred = np.where(pred >= 0, 1, -1)
        #################
        return pred
    
    
    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of simple randomforest classifier's decision.
            
        """
        ### CODE HERE ###
        # Initialize array for predictions
        pred = np.zeros(X.shape[0])
        for classifier_info in self.classifiers:
            # Get feature indices and classifier for each tree
            feature_indices = classifier_info['feature_indices']
            classifier = classifier_info['classifier']
            # Make predictions using only selected features
            X_features = X[:, feature_indices]
            prediction_i = classifier.predict(X_features)
            pred += prediction_i
        
        # Calculate probability values using sigmoid function
        sigmoid_values = 1 / (1 + np.exp(-pred))
        one_minus_sigmoid_values = 1 - sigmoid_values
        # Return probability values for each class
        proba = np.vstack((one_minus_sigmoid_values, sigmoid_values)).T
        #################
        return proba

        
        
def compute_each_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies of each classifier.
        
        Args:
            classifier_list: list of dictionary. RandomForest classifiers with feature indices.
            X_train: (N, d) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', d) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
            
    """
    acc_train = []
    acc_test = []

    for i in range(len(classifier_list)):
    
        ### CODE HERE ###
        # Get current classifier and its feature indices
        classifier_info = classifier_list[i]
        feature_indices = classifier_info['feature_indices']
        classifier = classifier_info['classifier']

        # Create train/test data using only selected features
        X_train_features = X_train[:, feature_indices]
        X_test_features = X_test[:, feature_indices]

        # Make predictions
        train_prediction_i = classifier.predict(X_train_features)
        test_prediction_i = classifier.predict(X_test_features)

        # Calculate accuracy
        accuracy_train_i = np.mean(train_prediction_i == y_train)
        accuracy_test_i = np.mean(test_prediction_i == y_test)

        # Store accuracies
        acc_train.append(accuracy_train_i)
        acc_test.append(accuracy_test_i)
        #################
            
    return acc_train, acc_test        
        
        
    
class AdaBoost:
    """AdaBoost class"""
    
    def __init__(self, num_estimators):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int. The number of decision stumps
                error_history: list. List of weighted error history.
                classifiers: list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[0]) is the dictionary denoted as classifier.
                             The classifier has key 'coefficient' and 'classifier'. The values are the coefficient 
                             for that classifier and the Decsion stump classifier.
        
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.error_history = []
        self.classifiers = []
        
    
    def fit(self, X, y):
        """
        Description:
            Build classifiers from the training set.
               
        Args:
            X: (N, d) numpy array. Training/testing samples.
            y: (N,) numpy array. Ground truth classes.
            
        Returns:
            
            
        """
        self.X = X
        self.y = y
        # initialize the data weight
        ### CODE HERE ###
        self.data_weight = np.ones(len(y)) / len(y)
        #################
        assert self.data_weight.shape == self.y.shape
        self.boosting()
        
    
    def boosting(self):
        """
        Description:
            Boosting - Build adaboost classifier. Follow the procedures described in .ipynb file.
            Use 'DecisionTreeClassifier(max_depth=1)' as a decision stump.
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###
        N = self.X.shape[0]
        for i in range(self.num_estimator):
            if (i==0):
                # Initialize first weak learner without sampling
                f_t = DecisionTreeClassifier(max_depth=1)
                f_t.fit(self.X, self.y)
                pred = f_t.predict(self.X)
            else:
                # Sample data points based on weights
                selected_indices = np.random.choice(np.arange(N), size=N, replace=True, p=self.data_weight)
                # Train weak learner on weighted samples
                f_t = DecisionTreeClassifier(max_depth=1)
                f_t.fit(self.X[selected_indices], self.y[selected_indices])
                pred = f_t.predict(self.X)
            
            # Calculate weighted error
            weighted_error = np.sum(self.data_weight[pred!=self.y])
            self.error_history.append(weighted_error)
            
            # Compute classifier weight
            w_t = self.compute_classifier_coefficient(weighted_error)

            # Store weak classifier and its weight
            classifier_info = {'classifier': f_t, 'coefficient': w_t}
            self.classifiers.append(classifier_info)
            
            # Update and normalize sample weights
            self.data_weight = self.update_weight(pred, w_t)
            self.data_weight = self.normalize_weight()
        #################
    
    
    def compute_classifier_coefficient(self, weighted_error):
        """
        Description:
            Compute the coefficient for classifier
        
        Args:
            weighted_error: numpy float. Weighted error for the classifier.
            
        Returns:
            coefficient: numpy float. Coefficient for classifier.
            
        """
        ### CODE HERE ###
        # Prevent division by zero or log(0)
        eps = 1e-10
        weighted_error = np.clip(weighted_error, eps, 1-eps)
        # Calculate classifier weight: alpha = 0.5 * ln((1-error)/error)
        coefficient = 0.5 * np.log((1 - weighted_error) / weighted_error)
        #################
        return coefficient
        
        
    def update_weight(self, pred, coefficient):
        """
        Description:
            Update the data weight. 
        
        Args:
            pred: (N, ) numpy array. Prediction of the weak classifier in one step.
            coefficient: numpy float. Coefficient for classifier.
            
        Returns:
            weight: (N, ) numpy array. Updated data weight.
            
        """
        ### CODE HERE ###
        # Update weights: w_i = w_i * exp(-alpha * y_i * h(x_i))
        weight = self.data_weight * np.exp(-coefficient * self.y * pred)
        #################
        return weight
        
        
    def normalize_weight(self):
        """
        Description:
            Normalize the data weight
        
        Args:
            
        Returns:
            weight: (N, ) numpy array. Normalized data weight.
            
        """
        ### CODE HERE ###
        # Normalize weights to sum to 1
        weight = self.data_weight / np.sum(self.data_weight)
        #################
        return weight
    
    
    def predict(self, X):
        """
        Description:
            Predict the target variables (Adaboosts' final prediction). Use the attribute classifiers.
            
            Note that item of classifiers list should be a dictionary like below
                self.classfiers[0] : classifier,  (dict)
                
            The dictionary {key: value} is composed,
                classifier : {'coefficient': (coefficient value),
                              'classifier' : (decision stump classifier)}
        
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of adaboost classifier. Output values are of 1 or -1.
            
        """
        ### CODE HERE ###
        # Initialize array for weighted predictions
        predictions = np.zeros(X.shape[0])
        for clf_dict in self.classifiers:
            clf = clf_dict['classifier']
            coef = clf_dict['coefficient']
            predictions += coef * clf.predict(X)
        
        # Apply sign function for final prediction
        pred = sign(predictions)
        #################
        return pred
    
    
    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class using sigmoid function. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of adaboost classifier's decision.
            
        """
        ### CODE HERE ###
        # Calculate weighted sum of predictions
        pred = np.zeros(X.shape[0])
        for classifier_info in self.classifiers:
            coefficient = classifier_info['coefficient']
            classifier = classifier_info['classifier']
            prediction_i = classifier.predict(X)
            pred += coefficient * prediction_i

        # Convert to probabilities using sigmoid function
        sigmoid_values = 1 / (1 + np.exp(-pred))
        one_minus_sigmoid_values = 1 - sigmoid_values
        # Return probability values for each class
        proba = np.vstack((one_minus_sigmoid_values, sigmoid_values)).T
        #################
        return proba
        
        
    
def compute_staged_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies over stages.
        
        Args:
            classifier_list: list of dictionary. Adaboost classifiers with coefficients.
            X_train: (N, d) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', d) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
                i.e, acc_train[40] =  $\hat{\mathbf{y}}=\text{sign} \left( \sum_{t=1}^{40} \hat{w_t} f_t(\mathbf{x}) \right)$
            
    """
    acc_train = []
    acc_test = []

    for i in range(len(classifier_list)):
    
        ### CODE HERE ###
        # Initialize cumulative predictions for current stage
        cumulative_train_scores = np.zeros(y_train.shape[0])
        cumulative_test_scores = np.zeros(y_test.shape[0])

        # Accumulate predictions from all classifiers up to current stage
        for stage in range(0, i+1):
            classifier_info = classifier_list[stage]
            weight = classifier_info['coefficient']
            weak_learner = classifier_info['classifier']

            cumulative_train_scores += (weight * weak_learner.predict(X_train))
            cumulative_test_scores += (weight * weak_learner.predict(X_test))

        # Calculate final predictions (apply sign function)
        final_train_predictions = np.where(cumulative_train_scores >= 0, 1, -1)
        final_test_predictions = np.where(cumulative_test_scores >= 0, 1, -1)

        # Calculate training accuracy
        train_correct_count = np.sum(final_train_predictions == y_train)
        train_accuracy = train_correct_count / len(y_train)

        # Calculate test accuracy
        test_correct_count = np.sum(final_test_predictions == y_test)
        test_accuracy = test_correct_count / len(y_test)

        # Store accuracies for current stage
        acc_train.append(train_accuracy)
        acc_test.append(test_accuracy)
        #################
            
    return acc_train, acc_test
    
    

def get_precision(y_pred, y_true):
    """
        Description:
            Compute the precision of the classifier.
        
        Args:
            y_pred: (N, ) numpy array. Prediction of the classifier.
            y_true: (N, ) numpy array. Ground truth classes.
        Returns:
            precision: float. Precision of the classifier.
            
    """
    ### CODE HERE ###
    # True Positives (TP): Correctly predicted positive cases
    TP = np.sum((y_pred == 1) & (y_true == 1))
    # False Positives (FP): Incorrectly predicted positive cases
    FP = np.sum((y_pred == 1) & (y_true == -1))
    
    # Prevent division by zero
    if (TP + FP) == 0:
        return 0.0
        
    precision = TP / (TP + FP)
    #################
    return precision

def get_recall(y_pred, y_true):
    """
        Description:
            Compute the recall of the classifier.
        
        Args:
            y_pred: (N, ) numpy array. Prediction of the classifier.
            y_true: (N, ) numpy array. Ground truth classes.
        Returns:
            recall: float. Recall of the classifier.
            
    """
    ### CODE HERE ###
    # True Positives (TP): Correctly predicted positive cases
    TP = np.sum((y_pred == 1) & (y_true == 1))
    # False Negatives (FN): Incorrectly predicted negative cases
    FN = np.sum((y_pred == -1) & (y_true == 1))
    
    # Prevent division by zero
    if (TP + FN) == 0:
        return 0.0
        
    recall = TP / (TP + FN)
    #################
    return recall

def get_f1_score(y_pred, y_true):
    """
        Description:
            Compute the F1 score of the classifier.
        
        Args:
            y_pred: (N, ) numpy array. Prediction of the classifier.
            y_true: (N, ) numpy array. Ground truth classes.
        Returns:
            f1_score: float. F1 score of the classifier.
            
    """
    ### CODE HERE ###
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    
    # Prevent division by zero
    if precision + recall == 0:
        return 0.0
        
    f1_score = 2 * (precision * recall) / (precision + recall)
    #################
    return f1_score


def plot_precision_recall_curve(y_proba, y_true):
    """
    Description:
        Compute and plot the precision-recall curve of the classifier.

    Args:
        y_proba: (N, 2) numpy array. Probability scores for each class (column 0 for class -1, column 1 for class 1).
        y_true: (N,) numpy array. Ground truth classes.

    Returns:
        precision_list: list. Precision of the classifier at each threshold.
        recall_list: list. Recall of the classifier at each threshold.
    """

    # Extract probabilities for the positive class (class 1)
    prob_pos = y_proba[:, 1]
    
    ### CODE HERE ###
    thresholds = np.sort(np.unique(prob_pos))
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        y_pred = np.where(prob_pos >= threshold, 1, -1)
        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        precision_list.append(precision)
        recall_list.append(recall)
    #################
    return precision_list, recall_list

