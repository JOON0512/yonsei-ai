from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    '''
    Dataset class
    '''
    def __init__(self, data_path):
        '''
        Description:
            Load a csv file using 'load_data' method.
            Convert the dataframe to numpy array.
        Args:
            data_path (str): Path of the csv file.
        Returns:
        '''
        assert type(data_path)==str, 'The type of data_path has to be string'
        ### CODE HERE ###
        self.dataframe = pd.read_csv(data_path)
        self.features = self.dataframe.columns.tolist()  # Store features as an attribute
        #################
    
    def load_data(self):
        '''
        Description:
            Load the csv file.
            Print the head of data.
            Gather names of the features & values of the data.
        Args:
        Returns:
            features (list): The names of the features.
            data (numpy array): Data.
        '''
        ### CODE HERE ###
        features = self.dataframe.columns.tolist()  # Get feature names
        data = self.dataframe.values  # Convert DataFrame values to a numpy array
        #################
        return features, data
    
    def gather_data(self,X,y,num_of_data):
        '''
        Description:
            Gather the first #(num_of_data) of the X,y dataset
        Args:
            X (numpy array): Input data
            y (numpy arary): Target (feature name: 'ViolentCrimesPerPop')
            num_of_data (int): Number of data that should be sliced.
        Returns:
            X (numpy array): Sliced input data
            y (numpy arary): Sliced target 
        '''
        X=X[:num_of_data,:]
        y=y[:num_of_data]
        return X,y
    
    def parse_data(self, features):
        '''
        Description:
            Parse the data using 'features'
            Initialize bias 1 and concatenate it in front of input data. (10.12 update)
        Args:
            features (list): The names of feature we use.
        Returns:
            X (numpy array): Input data
            y (numpy arary): Target (feature name: 'ViolentCrimesPerPop')
        '''
        assert type(features)==list, 'The type of feature_names has to be list'
        assert all([isinstance(feature, str) for feature in features]), 'The element of features has to be string'
        
        ### CODE HERE ###
        y = self.dataframe['ViolentCrimesPerPop'].values  # Extract the target variable (y) from the specified column
        # Select specified features, add a bias term, and convert to numpy array
        X = self.dataframe[features].copy()  # Select only the specified feature columns
        y_column = ['ViolentCrimesPerPop']  # Specify the target column so it can be excluded from features
        X_1 = self.dataframe.drop(columns=y_column).copy()  # Create a new DataFrame, excluding the target column
        X_1.insert(0, 'Bias', 1)  # Insert a 'Bias' column with all values set to 1 at the beginning of X_1
        X = X_1.values  # Convert to a numpy array for further processing
        #################
        return X, y


class Regressor:
    '''
    Regressor class
    '''
    def __init__(self, tau, dim):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        ### CODE HERE ###
        self.tau = tau
        self.dim = dim
        #################
    
    def initialize_weight(self):
        '''
        Description:
            Initialize the weight of the model using normal distribution.
        Args:
        Returns:
            weight (numpy array): Initialized weight.
        '''
        np.random.seed(0)
        ### CODE HERE ###
        self.weight = np.random.normal(0, 1, self.dim)
        return self.weight
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        '''
        pass
    
    def learning_with_closed_form(self, X, y):
        '''
        Description:
            Set attribute weight_closed_form using the closed form solution.
        '''
        pass
    
    def learning_with_gradient_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_gradient_descent using initialize_weight method.
            Update weight_gradient_descent until convergence using gradient descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        '''
        pass
    
    def loss_history(self):
        '''
        Description:
            Plot the history of the RSS loss.
        Args:
        Returns:
        '''
        ### CODE HERE ###
        # Plot self.losses to display the training loss over iterations
        # plt.figure(figsize=(8, 5))
        plt.title("RSS Loss Over Iterations")
        plt.xlabel("Iterations")  # Label the axes and set the title
        plt.ylabel("RSS Loss")
        plt.plot(self.losses, color='red', linestyle='--', marker='o', label="RSS Loss")
        # plt.legend()  # Add a legend to the plot
        plt.grid(True)  # Display grid for better readability
        plt.show()  # Display the plot
        #################


class LinearRegressor(Regressor):
    '''
    Linear regressor class
    '''
    def __init__(self, tau, dim, lr=1e-5):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        assert type(lr)==float, 'The type of lr has to be float'
        ### CODE HERE ###
        self.tau = tau   # convergence threshold
        self.dim = dim   # number of features
        self.lr = lr     # learning rate
        self.weight_coordinate_descent = np.zeros(dim)  # initialize weights as zero
        self.weight_gradient_descent = np.random.normal(0, 1, dim)
        self.losses = []
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        ### CODE HERE ###
        # Initialize weights using normal distribution
        self.weight_coordinate_descent = self.initialize_weight()
        # Initialize tracking for losses and initial weights
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]

        max_value = float('inf')  # Initial value to ensure loop starts
        previous_weight = np.zeros(self.dim)  # To track weight changes

        # Loop until convergence criterion is met
        while max_value > self.tau:
            previous_weight = self.weight_coordinate_descent.copy()
            
            # Update each weight coordinate
            for j in range(self.dim):
                # Calculate residual without the effect of the current weight[j]
                residual = y - (X @ self.weight_coordinate_descent) + (X[:, j] * self.weight_coordinate_descent[j])
                
                # Update weight[j] based on the residual
                self.weight_coordinate_descent[j] = np.dot(X[:, j], residual) / np.sum(X[:, j] ** 2)
            
            # Calculate and record the RSS loss after each full iteration
            current_loss = RSSloss(X, y, self.weight_coordinate_descent)
            self.losses.append(current_loss)
            
            # Convergence check using infinity norm
            difference = previous_weight - self.weight_coordinate_descent
            max_value = np.max(np.abs(difference))
        return self.losses
        #################
    
    def learning_with_closed_form(self, X, y):
        '''
        Description:
            Set attribute weight_closed_form using the closed form solution.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        ### CODE HERE ###
        X_transpose = X.T
        self.weight_closed_form = np.linalg.inv(X_transpose @ X) @ X_transpose @ y  # np.linalg.inv() computes the matrix inverse
        #################
    
    def learning_with_gradient_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_gradient_descent using initialize_weight method.
            Update weight_gradient_descent until convergence using gradient descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        # Initialize weights and set the initial conditions
        self.weight_gradient_descent = self.initialize_weight()
        self.losses = [RSSloss(X, y, self.weight_gradient_descent)]
        current_w = deepcopy(self.weight_gradient_descent)

        # Loop until the convergence condition is met
        while True:
            prev_w = deepcopy(current_w)
            partial = -2 * np.dot(np.transpose(X), y - np.dot(X, prev_w))
            current_w = prev_w - self.lr * partial
            if max(np.absolute(prev_w - current_w)) < self.tau:
                self.weight_gradient_descent = deepcopy(prev_w)
                break
            self.losses.append(RSSloss(X, y, current_w))
            # Calculate predictions
            # y_pred = X @ self.weight_gradient_descent
            # # Compute the gradient
            # gradient = -2 * X.T @ (y - y_pred)
            # # Update weights
            # new_weight = self.weight_gradient_descent - self.lr * gradient
            # # Calculate the RSS loss and store it
            # current_loss = RSSloss(X, y, new_weight)
            # self.losses.append(current_loss)
            # # Compute the infinity norm of the weight change
            # max_change = np.max(np.abs(new_weight - self.weight_gradient_descent))
            # # Update the weight for the next iteration
            # self.weight_gradient_descent = new_weight
        # return self.losses
        #################


class RidgeRegressor(Regressor):
    '''
    Ridge regressor class
    '''
    def __init__(self, dim, tau=1e-3, lam=0, lr=1e-5):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
            lam (float or int): Regularization parameter.
            lr (float): Learning rate.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        assert type(lam)==float or type(lam)==int, 'The type of lam has to be float or int'
        assert type(lr)==float, 'The type of lr has to be float'
        ### CODE HERE ###
        
        self.tau = tau
        self.dim = dim
        self.lam = lam
        self.lr = lr
        # self.weight_coordinate_descent = np.zeros(dim)
        # self.weight_gradient_descent = np.zeros(dim)
        # self.losses = []
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        ### CODE HERE ###
        self.weight_coordinate_descent = np.random.normal(0, 1, self.dim)  # Initialize weights
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]

        feature_squares_sum = np.sum(np.square(X), axis=0)
        current_weights = deepcopy(self.weight_coordinate_descent)

        while True:
            previous_weights = deepcopy(current_weights)
            for j in range(self.dim):                                                   
                partial_residual = y - X @ current_weights + X[:, j] * current_weights[j]
                rho_j = X[:, j].T @ partial_residual
                
                # Ridge regularization: add lambda for non-bias terms
                if j == 0:
                    current_weights[j] = rho_j / feature_squares_sum[j]  # No regularization for bias
                else:
                    current_weights[j] = rho_j / (feature_squares_sum[j] + self.lam)
            if np.max(np.abs(previous_weights - current_weights)) < self.tau:
                self.weight_coordinate_descent = deepcopy(current_weights)
                break

            self.losses.append(RSSloss(X, y, current_weights))

        # max_value = float('inf')  # Initial value to ensure loop starts
        # previous_weight = np.zeros(self.dim)  # To track weight changes

        # # Loop until convergence criterion is met
        # while max_value > self.tau:
        #     previous_weight = self.weight_coordinate_descent.copy()
            
        #     # Update each weight coordinate
        #     for j in range(self.dim):
        #         # Calculate residual without the effect of the current weight[j]
        #         residual = y - (X @ self.weight_coordinate_descent) + (X[:, j] * self.weight_coordinate_descent[j])
                
        #         # Update weight[j] based on the residual
        #         self.weight_coordinate_descent[j] = np.dot(X[:, j], residual) / np.sum(X[:, j] ** 2)
            
        #     # Calculate and record the RSS loss after each full iteration
        #     current_loss = RSSloss(X, y, self.weight_coordinate_descent)
        #     self.losses.append(current_loss)
            
        #     # Convergence check using infinity norm
        #     difference = previous_weight - self.weight_coordinate_descent
        #     max_value = np.max(np.abs(difference))
        # return self.losses
        #################
    
    def learning_with_closed_form(self, X, y):
        '''
        Description:
            Set attribute weight_closed_form using the closed form solution.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        ### CODE HERE ###
        I = np.identity(X.shape[1])
        I[0,0] = 0
        regularization_matrix = self.lam * I                                  # Regularization term

        inverse = np.linalg.inv(np.dot(np.transpose(X), X) + regularization_matrix)
        self.weight_closed_form = np.dot(np.dot(inverse, np.transpose(X)), y)          # weight 계산, w_hat = (X^T * X + lambda * I)^(-1) * X^T * y
        self.losses.append(RSSloss(X, y, self.weight_closed_form))                          # 최종 loss 계산 후 저장
        #################
    
    def learning_with_gradient_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_gradient_descent using initialize_weight method.
            Update weight_gradient_descent until convergence using gradient descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        ### CODE HERE ###

        self.weight_gradient_descent = self.initialize_weight()                            
        self.losses = [RSSloss(X, y, self.weight_gradient_descent)]                        
        while True:
            previous_weights = deepcopy(self.weight_gradient_descent)
            residual = y - np.dot(X, previous_weights)
            partial = -2 * np.dot(X.T, residual) + 2 * self.lam * (self.weight_gradient_descent)
            partial[0] -= 2 * self.lam * (self.weight_gradient_descent[0])

            self.weight_gradient_descent -= self.lr * partial

            self.losses.append(RSSloss(X, y, self.weight_gradient_descent))                               
            if max(np.absolute(self.weight_gradient_descent - previous_weights)) <= self.tau:                         
                break

        # self.weight_gradient_descent = self.initialize_weight()
        # self.losses = [RSSloss(X, y, self.weight_gradient_descent)]
        # current_weights = deepcopy(self.weight_gradient_descent)
        
        # while True:
        #     previous_weights = deepcopy(current_weights)
        #     gradient = -2 * X.T @ (y - X @ current_weights) + 2 * self.lam * current_weights
        #     current_weights = (1 - 2 * self.lr * self.lam) * previous_weights - self.lr * gradient
            
        #     if np.max(np.abs(previous_weights - current_weights)) < self.tau:
        #         self.weight_gradient_descent = deepcopy(current_weights)
        #         break
            
        #     self.losses.append(RSSloss(X, y, current_weights))  # Store loss for each iteration
        #################


class LassoRegressor(Regressor):
    '''
    Lasso regressor class
    '''
    def __init__(self, dim, tau=1e-3, lam=0):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
            lam (float or int): Regularization parameter.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        assert type(lam)==float or type(lam)==int, 'The type of lam has to be float or int'
        ### CODE HERE ###
        self.dim = dim
        self.tau = tau
        self.lam = lam
        self.weight = self.initialize_weight()
        #################
    
    def thresholding(self, rho, lam):
        '''
        Description:
            Implement the thresholding function.
        Args:
            rho (float): The value to be thresholded.
            lam (float): Regularization parameter.
        Returns:
            out (float): The thresholded value.
        '''
        ### CODE HERE ###
        if rho < (-lam / 2):
            return (rho + lam / 2)    
        elif rho > (lam / 2):
            return (rho - lam / 2)
        else:
            return 0
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
    ### CODE HERE ###
        # Initialize weights and initial loss
        self.weight_coordinate_descent = self.initialize_weight()
        initial_loss = RSSloss(X, y, self.weight_coordinate_descent)
        self.losses = [initial_loss]

        # Precompute the sum of squares for each feature column in X
        z = np.einsum('ij,ij->j', X, X)
        current_weights = deepcopy(self.weight_coordinate_descent)

        # Begin coordinate descent iterations
        while True:
            # Track the previous weights to check for convergence
            previous_weights = current_weights.copy()
            
            # Update each weight in turn
            for j in range(self.dim):
                # Calculate the partial residual for the current feature
                partial_residual = y - X @ current_weights + X[:, j] * current_weights[j]
                rho_j = X[:, j].T @ partial_residual
                
                # Update the weight with or without regularization
                if j == 0:  # No regularization for intercept
                    current_weights[j] = rho_j / z[j]
                else:  # Apply thresholding with regularization
                    current_weights[j] = self.thresholding(rho_j, self.lam) / z[j]
            
            # Check for convergence using the maximum absolute change in weights
            weight_change = np.abs(previous_weights - current_weights)
            if np.max(weight_change) < self.tau:
                self.weight_coordinate_descent = deepcopy(current_weights)
                break
            
            updated_loss = RSSloss(X, y, current_weights)
            self.losses.append(updated_loss)
        #################


class ElasticNetRegressor(Regressor):
    '''
    ElasticNet regressor class
    '''
    def __init__(self, dim, tau=1e-3, lam=0, alpha=0.5):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
            lam (float or int): Regularization parameter.
            alpha (float): Elastic net mixing parameter (0 = ridge, 1 = lasso).
        Returns:
        '''
        assert type(tau) == float, 'The type of tau has to be float'
        assert type(dim) == int, 'The type of dim has to be integer'
        assert type(lam) == float or type(lam) == int, 'The type of lam has to be float or int'
        assert type(alpha) == float, 'The type of alpha has to be float'
        ### CODE HERE ###
        self.dim = dim
        self.tau = tau
        self.lam = lam
        self.alpha = alpha
        self.weight_coordinate_descent = self.initialize_weight()
        self.losses = []
        #################
    
    def thresholding(self, rho, lam):
        '''
        Description:
            Implement the thresholding function for Lasso regularization.
        Args:
            rho (float): The value to be thresholded.
            lam (float): Regularization parameter.
        Returns:
            out (float): The thresholded value.
        '''
        ### CODE HERE ###
        threshold = self.alpha * lam / 2
        if rho < -threshold:
            return rho + threshold
        elif rho > threshold:
            return rho - threshold
        else:
            return 0
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
    ### CODE HERE ###
        # Initialize weights and initial loss
        self.weight_coordinate_descent = self.initialize_weight()
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]

        lam_alpha = self.alpha * self.lam
        lam_l2 = (1 - self.alpha) * self.lam
        z = np.sum(X ** 2, axis = 0)

        loss = RSSloss(X, y, self.weight_coordinate_descent)
        self.losses = [loss]

        residual = y - X @ self.weight_coordinate_descent

        
        while True:
            weight_prev = np.copy(self.weight_coordinate_descent)
            
            for j in range(self.dim):
                residual += X[:, j] * self.weight_coordinate_descent[j]

                rho = np.dot(X[:, j], residual)
                if j == 0:
                    self.weight_coordinate_descent[j] = rho / z[j]
                else:
                    self.weight_coordinate_descent[j] = self.thresholding(rho, lam_alpha) / (z[j] + lam_l2)
                    
                residual -= X[:, j] * self.weight_coordinate_descent[j]

            loss = RSSloss(X, y, self.weight_coordinate_descent)
            self.losses.append(loss)

            if np.max(np.abs(self.weight_coordinate_descent - weight_prev)) <= self.tau:
                break 


def RSSloss(X, y, weight):
    '''
    Description:
        Calculate the RSS loss (error).
    Args:
        X (numpy array): Input data
        y (numpy array or float): Target data
        weight (numpy array): Weight of the model
    Returns:
        loss (float): RSS loss (error).
    '''
    ### CODE HERE ###
    y_pred = X @ weight
    residuals = y - y_pred
    loss = np.sum(residuals ** 2)
    return loss
    #################
