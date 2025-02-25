import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W = None  # Weights
        self.b = None  # Bias
        self.loss_history = []  # Records validation loss for each batch

    def fit(self, X, y, batch_size, regularization=0, max_epochs=100, patience=3):
        """
        Trains the model using mini-batch gradient descent with early stopping.
        
        Parameters:
          X             : Input data (n_samples x n_features)
          y             : Target values (n_samples) or (n_samples x 1)
          batch_size    : Batch size for gradient descent
          regularization: L2 regularization factor (default 0)
          max_epochs    : Maximum number of epochs to train
          patience      : Number of consecutive batches with no improvement 
                          on validation loss before stopping
        """
        X = np.array(X)
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        # Split training data into training and validation sets (90%/10% split)
        val_size = int(0.1 * n_samples)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Initialize weights and bias
        self.W = np.zeros((n_features, n_outputs))
        self.b = np.zeros((1, n_outputs))

        best_W = self.W.copy()
        best_b = self.b.copy()
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(max_epochs):
            # Shuffle training data at the beginning of each epoch
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            n_batches = int(np.ceil(X_train_shuffled.shape[0] / batch_size))
            
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, X_train_shuffled.shape[0])
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward pass: compute predictions
                y_pred = np.dot(X_batch, self.W) + self.b

                # Compute error and gradients
                error = y_pred - y_batch
                grad_W = (np.dot(X_batch.T, error) / X_batch.shape[0]) + regularization * self.W
                grad_b = np.sum(error, axis=0, keepdims=True) / X_batch.shape[0]

                # Update parameters
                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b

                # Evaluate on the validation set
                y_val_pred = self.predict(X_val)
                val_loss = np.mean((y_val_pred - y_val) ** 2)
                self.loss_history.append(val_loss)

                # Early stopping: if validation loss improves, save model; else increment counter
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_W = self.W.copy()
                    best_b = self.b.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}, batch {i+1}")
                        self.W = best_W
                        self.b = best_b
                        return

        # After training, set the model parameters to the best observed
        self.W = best_W
        self.b = best_b

    def predict(self, X):
        """
        Computes predictions for the input data X.
        """
        X = np.array(X)
        return np.dot(X, self.W) + self.b

    def score(self, X, y):
        """
        Computes the mean squared error (MSE) between predictions and target values.
        
        Parameters:
          X : Input data
          y : True target values
          
        Returns:
          MSE : Mean squared error
        """
        y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

# Demonstration using the iris dataset
if __name__ == '__main__':
    # Load iris dataset
    iris = load_iris()
    X = iris.data  # shape: (n_samples, 4) containing all features
    
    # For regression, choose a combination of features.
    # Example: Predict petal width (feature index 3) using petal length (index 2) and sepal width (index 1).
    X_reg = X[:, [2, 1]]
    y_reg = X[:, 3]

    # Split data into training and testing sets (10% reserved for testing)
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.1, random_state=42)

    # Create and train the LinearRegression model
    model = LinearRegression(learning_rate=0.01)
    model.fit(X_train, y_train, batch_size=32, regularization=0.01, max_epochs=100, patience=3)

    # Evaluate the model on the training and test sets
    train_mse = model.score(X_train, y_train)
    test_mse = model.score(X_test, y_test)
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Optional: Plot the validation loss history
    plt.plot(model.loss_history)
    plt.xlabel('Training Step')
    plt.ylabel('Validation MSE')
    plt.title('Validation Loss History')
    plt.show()
