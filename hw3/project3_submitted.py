import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.utils import check_random_state


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_hdim3, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
            W4: Fourth layer weights
            b4: Fourth layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons in the hidden layer H1.
            nn_hdim2: (int) The number of neurons in the hidden layer H2.
            nn_hdim3: (int) The number of neurons in the hidden layer H3.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random": # random initialization (standard normal, bias is 0)
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_hdim3)
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.random.randn(nn_hdim3, nn_output_dim)
            self.model['b4'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_hdim3))
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.ones((nn_hdim3, nn_output_dim))
            self.model['b4'] = np.zeros((1, nn_output_dim))

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
            
        """
        W1, b1 = self.model['W1'], self.model['b1']
        W2, b2 = self.model['W2'], self.model['b2']
        W3, b3 = self.model['W3'], self.model['b3']
        W4, b4 = self.model['W4'], self.model['b4']
        
        ### CODE HERE ###
        # First layer: Using ReLU activation
        temp_1 = X.dot(W1)           # break it into two lines for variation
        h1 = temp_1 + b1
        z1 = relu(h1)                # ReLU activation

        # Second layer: Using LeakyReLU activation
        temp_2 = z1.dot(W2)
        h2 = temp_2 + b2
        z2 = leakyrelu(h2)

        # Third layer: Using Tanh activation + skip connection with h1
        temp_3 = z2.dot(W3)
        h3 = temp_3 + b3
        z3 = tanh(h3 + h1)

        # Output layer: Using Sigmoid activation
        temp_4 = z3.dot(W4)
        h4 = temp_4 + b4
        y_hat = sigmoid(h4)
        y_hat = y_hat.reshape(X.shape[0],)
        ############################
        
        cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2,
                 'h3': h3, 'z3': z3, 'h4': h4, 'y_hat': y_hat}
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) Training labels (N, ) -> (N, 1)
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        """
        W1, b1 = self.model['W1'], self.model['b1']
        W2, b2 = self.model['W2'], self.model['b2']
        W3, b3 = self.model['W3'], self.model['b3']
        W4, b4 = self.model['W4'], self.model['b4']
        h1, z1 = cache['h1'], cache['z1']
        h2, z2 = cache['h2'], cache['z2']
        h3, z3 = cache['h3'], cache['z3']
        h4, y_hat = cache['h4'], cache['y_hat']
        
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        
        ### CODE HERE ###
        # Cross-entropy partial derivative wrt y_hat
        partial_y_hat = - (y / y_hat) + (1 - y)/(1 - y_hat)

        # Sigmoid derivative for output layer
        dh4 = partial_y_hat * (y_hat * (1 - y_hat))
        db4 = np.sum(dh4, axis=0)
        dW4 = np.dot(z3.T, dh4) + 2 * L2_norm * W4
        dz3 = np.dot(dh4, W4.T)

        # Tanh derivative for third hidden layer (plus skip connection)
        dtanh_factor = 1 - tanh(h1 + h3)**2
        dh3 = dz3 * dtanh_factor
        db3 = np.sum(dh3, axis=0)
        dW3 = z2.T.dot(dh3) + 2 * L2_norm * W3
        dz2 = dh3.dot(W3.T)

        # LeakyReLU derivative for second hidden layer
        alpha = 0.01
        dh2 = dz2 * np.where(h2 > 0, 1, alpha)
        db2 = np.sum(dh2, axis=0)
        dW2 = z1.T.dot(dh2) + 2 * L2_norm * W2
        dz1 = dh2.dot(W2.T)

        # ReLU derivative for first hidden layer (plus skip)
        relu_mask_1 = (h1 > 0).astype(float)
        dh1 = dz1 * relu_mask_1 + dh3
        db1 = np.sum(dh1, axis=0)
        dW1 = X.T.dot(dh1) + 2 * L2_norm * W1
        ############################
        
        grads = dict()
        grads['dy_hat'] = partial_y_hat
        grads['dh4'] = dh4
        grads['dW4'] = dW4
        grads['db4'] = db4
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    
    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Calculate total loss including L2 regularization for both weights and biases
        """
        W1, b1 = self.model['W1'], self.model['b1']
        W2, b2 = self.model['W2'], self.model['b2']
        W3, b3 = self.model['W3'], self.model['b3']
        W4, b4 = self.model['W4'], self.model['b4']
        
        loss = -np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
        reg_loss = (np.sum(W1**2) + np.sum(b1**2) +
                    np.sum(W2**2) + np.sum(b2**2) +
                    np.sum(W3**2) + np.sum(b3**2) +
                    np.sum(W4**2) + np.sum(b4**2))
        
        total_loss = loss + L2_norm * reg_loss
        return total_loss
        

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for it in range(epoch):
            ### CODE HERE ###
            # Forward pass
            y_hat, cache = self.forward_propagation(X_train)

            # Compute loss
            loss_val = self.compute_loss(y_hat, y_train, L2_norm=L2_norm)

            # Backward pass
            grad = self.back_propagation(cache, X_train, y_train, L2_norm)

            # Parameter update
            self.model['W1'] -= learning_rate * grad['dW1']
            self.model['b1'] -= learning_rate * grad['db1']
            self.model['W2'] -= learning_rate * grad['dW2']
            self.model['b2'] -= learning_rate * grad['db2']
            self.model['W3'] -= learning_rate * grad['dW3']
            self.model['b3'] -= learning_rate * grad['db3']
            self.model['W4'] -= learning_rate * grad['dW4']
            self.model['b4'] -= learning_rate * grad['db4']
            #################   

            if (it+1) % 1000 == 0:
                loss_history.append(loss_val)
                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss_val}")
 
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        inference, _ = self.forward_propagation(X)
        pred = np.rint(inference).astype(int)
        return pred
        #################  


def tanh(x):
    ### CODE HERE ###
    # Rewriting the same formula in a slightly different style
    pos = np.exp(x)
    neg = np.exp(-x)
    x = (pos - neg) / (pos + neg)
    #################  
    return x


def relu(x):
    ### CODE HERE ###
    # Using elementwise multiplication as an equivalent to np.maximum(0,x)
    mask = (x > 0).astype(float)
    x = x * mask
    ############################
    return x


def leakyrelu(x):
    ### CODE HERE ###
    alpha = 0.01
    x = np.where(x >= 0, x, alpha * x)
    ############################
    return x 


def sigmoid(x):
    ### CODE HERE ###
    # Slightly more explicit version
    denom = 1.0 + np.exp(-x)
    x = 1.0 / denom
    ############################
    return x


######################################################################################


class Linear(object):
    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear layer.
        
        Args:
            x: (numpy array) Array containing input data, of shape (N, D)
            w: (numpy array) Array of weights, of shape (D, M)
            b: (numpy array) Array of biases, of shape (M,)

        Returns: 
            out: (numpy array) output, of shape (N, M)
            cache: (tuple[numpy array]) Values needed to compute gradients
        """
        ### CODE HERE ###
        out = np.dot(x, w) + b
        cache = (x, w)
        #################
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for an linear layer.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: (numpy array) Gradient with respect to x, of shape (N, D)
            dw: (numpy array) Gradient with respect to w, of shape (D, M)
            db: (numpy array) Gradient with respect to b, of shape (M,)
        """
        ### CODE HERE ###
        x, w = cache
        dx = np.dot(dout, w.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        #################  
        return dx, dw, db


class ReLU(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = np.maximum(0, x)
        cache = (x, out)
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, _ = cache
        dx = dout * (x > 0)
        #################  
        return dx


class LeakyReLU(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of LeakyReLUs.

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = np.where(x > 0, x, 0.01*x)
        cache = (x, out)
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of LeakyReLU.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, _ = cache
        dx = dout * (x > 0) + (0.01 * dout) * (x <= 0)
        #################  
        return dx


class Tanh(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Tanh.

        Args:
            x: Input

        Returns:
            out: Output, array of the same shape as x
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = np.tanh(x)
        cache = (x, out)
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Tanh.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        x, out = cache
        dx = dout * (1 - out**2)
        #################  
        return dx


class Sigmoid(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Sigmoid.

        Args:
            x: Input

        Returns:
            out: Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        out = 1.0 / (1.0 + np.exp(-x))
        cache = (x, out)
        #################  
        return out, cache

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Sigmoid.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        _, out = cache 
        dx = dout * out * (1 - out)
        #################  
        return dx


class SigmoidWithBCEloss(object): 
    @staticmethod
    def forward(x, y=None):
        """
        if y is None, computes the forward pass for a layer of sigmoid with binary cross-entropy loss.
        Else, computes the loss for binary classification.
        Args:
            x: Input data
            y: Training labels or None 
       
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N,) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        y_hat, _ = Sigmoid.forward(x)
        y_hat = y_hat.reshape(-1)
        if y is None:
            return y_hat
        loss = -np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        cache = (y_hat, y)
        ################# 
        return loss, cache

    @staticmethod
    def backward(cache, dout=None):
        """
        Computes the loss and gradient for softmax classification.
        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        y_hat, y = cache
        y = y.reshape(-1, 1)
        y_hat = y_hat.reshape(-1, 1)
        
        ### CODE HERE ###
        dy_hat = -y / y_hat + (1 - y)/(1 - y_hat)
        dx = dy_hat * (y_hat * (1 - y_hat))
        ################# 
        return dx


class Dropout(object):
    @staticmethod
    def forward(x, dropout_rate, train=False):
        """
        Computes the forward pass for a layer of Dropout.
        
        Args:
            x: Input data
            dropout_rate: Probability of dropping a neuron active during dropout
            train: (bool) if True, perform dropout
        
        Returns:
            out: Output data
            cache: Values needed for backward pass
        """
        ### CODE HERE ###
        if train:
            mask = (np.random.rand(*x.shape) > dropout_rate)
            scale = 1.0 / (1.0 - dropout_rate)
            out = x * mask * scale
            cache = (mask, scale)
        else:
            out = x
            cache = None
        #################  
        return out, cache
    
    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Dropout.
        
        Args:
            cache: Values needed for backward pass
            dout: Upstream derivatives
        
        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        if cache is None:
            dx = dout
        else:
            mask, scale = cache
            dx = dout * mask * scale
        #################  
        return dx


class NeuralNetwork_module(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_hdim3, nn_output_dim, dropout_rate, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H2.
            nn_hdim3: (int) The number of neurons H3 in the hidden layer H3.
            nn_output_dim: (int) The number of classes C.
            dropout_rate: (float) Probability of dropping a neuron active during dropout
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_hdim3)
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.random.randn(nn_hdim3, nn_output_dim)
            self.model['b4'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_hdim3))
            self.model['b3'] = np.zeros((1, nn_hdim3))
            self.model['W4'] = np.ones((nn_hdim3, nn_output_dim))
            self.model['b4'] = np.zeros((1, nn_output_dim))
            
        self.dropout_rate = dropout_rate

    def forward(self, X, y=None, train=False):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        If y is None, returns just y_hat.
        Otherwise, returns (cache, loss).
        """

        W1, b1 = self.model['W1'], self.model['b1']
        W2, b2 = self.model['W2'], self.model['b2']
        W3, b3 = self.model['W3'], self.model['b3']
        W4, b4 = self.model['W4'], self.model['b4']
        cache = {}
        
        ### CODE HERE ###
        h1, cache['Linear_h1'] = Linear.forward(X, W1, b1)
        z1, cache['ReLU'] = ReLU.forward(h1)
        z1, cache['dropout1'] = Dropout.forward(z1, self.dropout_rate, train=train)

        h2, cache['Linear_h2'] = Linear.forward(z1, W2, b2)
        z2, cache['LeakyReLU'] = LeakyReLU.forward(h2)
        z2, cache['dropout2'] = Dropout.forward(z2, self.dropout_rate, train=train)

        h3, cache['Linear_h3'] = Linear.forward(z2, W3, b3)
        # skip connection with h1
        skip_in = h1 + h3
        z3, cache['Tanh'] = Tanh.forward(skip_in)
        z3, cache['dropout3'] = Dropout.forward(z3, self.dropout_rate, train=train)

        out, cache['Linear_h4'] = Linear.forward(z3, W4, b4)
        #################
        
        if y is None:
            # just do inference
            y_hat = SigmoidWithBCEloss.forward(out)
            return y_hat
        else:
            # training: compute BCE loss
            loss, cache['SigmoidWithBCEloss'] = SigmoidWithBCEloss.forward(out, y)
            return cache, loss
    
    def backward(self, cache, L2_norm=0.0):
        """
        Compute the gradients
        """
        dh4 = SigmoidWithBCEloss.backward(cache['SigmoidWithBCEloss'])

        ### CODE HERE ###
        dz3, dW4, db4 = Linear.backward(cache['Linear_h4'], dh4)
        dW4 += 2 * L2_norm * cache['Linear_h4'][1]

        dz3 = Dropout.backward(cache['dropout3'], dz3)
        dtanh = Tanh.backward(cache['Tanh'], dz3)
        dz2, dW3, db3 = Linear.backward(cache['Linear_h3'], dtanh)
        dW3 += 2 * L2_norm * cache['Linear_h3'][1]

        dz2 = Dropout.backward(cache['dropout2'], dz2)
        dLeaky = LeakyReLU.backward(cache['LeakyReLU'], dz2)
        dz1, dW2, db2 = Linear.backward(cache['Linear_h2'], dLeaky)
        dW2 += 2 * L2_norm * cache['Linear_h2'][1]

        dz1 = Dropout.backward(cache['dropout1'], dz1)
        dReLU = ReLU.backward(cache['ReLU'], dz1)
        # plus skip connection gradient
        dReLU += dtanh  
        _, dW1, db1 = Linear.backward(cache['Linear_h1'], dReLU)
        dW1 += 2 * L2_norm * cache['Linear_h1'][1]
        #################

        grads = {}
        grads['dW4'] = dW4
        grads['db4'] = db4
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        return grads

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True, momentum=0.0):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization
            momentum: (float) Scalar giving momentum strength for optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        if momentum == 0.0:
            optimizer = GradientDescent(learning_rate=learning_rate)
        else:
            optimizer = Momentum(learning_rate=learning_rate, rho=momentum)

        for it in range(epoch):
            ### CODE HERE ###
            cache, loss = self.forward(X_train, y_train, train=True)
            grads = self.backward(cache, L2_norm)
            self.model = optimizer.step(self.model, grads)
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)
                
                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)
                    
            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")

         
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        y_hat = self.forward(X, y=None, train=False)
        pred = (y_hat >= 0.5).astype(int)
        return pred
        #################  


class GradientDescent(object):
    def __init__(self, learning_rate=0.01):
        """
        Args:
            learning_rate: (float) Scalar giving learning rate for optimization
        Returns:
        """
        self.learning_rate = learning_rate
        
    def step(self, params:dict, grads:dict):
        """
        Args:
            params: (dict) Dictionary mapping parameter names to arrays of parameter values
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        Returns:
            params: (dict) Dictionary mapping parameter names to updated parameter values
        """
        ### CODE HERE ###
        for key in params:
            if 'd'+key in grads:
                params[key] -= self.learning_rate * grads['d'+key]
        #################
        return params


class Momentum(object):
    def __init__(self, learning_rate=0.01, rho=0.9):
        """
        Args:
            learning_rate: (float) Scalar giving learning rate for optimization
            rho: (float) Scalar giving momentum strength
        Returns:
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.velocity = {}
        
    def step(self, params:dict, grads:dict):
        """
        Args:
            params: (dict) Dictionary mapping parameter names to arrays of parameter values
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
        Returns:
            params: (dict) Dictionary mapping parameter names to updated parameter values
        """
        ### CODE HERE ###
        for key in params:
            grad_key = 'd' + key
            if grad_key in grads:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(params[key])
                self.velocity[key] = self.rho * self.velocity[key] + grads[grad_key]
                params[key] -= self.learning_rate * self.velocity[key]
        #################
        return params
