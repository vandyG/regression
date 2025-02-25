import numpy as np
import os
import pickle


class LogisticRegression:
    def __init__(self, learning_rate=0.01, multi_class="ovr"):
        """
        Initialize Logistic Regression model.

        Parameters:
        learning_rate (float): Learning rate for gradient descent
        multi_class (str): Multi-class classification strategy.
                          'ovr' for One-vs-Rest (default)
        """
        self.learning_rate = learning_rate
        self.multi_class = multi_class
        self.W = None
        self.b = None
        self.classes_ = None
        self.n_classes = None
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        """
        Sigmoid activation function

        Parameters:
        z (numpy.ndarray): Input data

        Returns:
        numpy.ndarray: Sigmoid activation result
        """
        # Clip z to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """
        Softmax activation function for multi-class classification

        Parameters:
        z (numpy.ndarray): Input data

        Returns:
        numpy.ndarray: Softmax probabilities
        """
        # Subtract max for numerical stability
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """
        Trains the logistic regression model using gradient descent.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features)
        y (numpy.ndarray): Target values of shape (n_samples,)
        batch_size (int): Size of each batch during training
        regularization (float): L2 regularization factor
        max_epochs (int): Maximum number of epochs to train
        patience (int): Number of epochs to wait before early stopping

        Returns:
        self: The trained model
        """
        n_samples, n_features = X.shape

        # Find unique classes
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        # Convert y to one-hot encoding for multi-class
        if self.n_classes > 2:
            y_onehot = np.zeros((n_samples, self.n_classes))
            for i, cls in enumerate(self.classes_):
                y_onehot[:, i] = (y == cls).astype(int)
            y_encoded = y_onehot
        else:
            # Binary classification - no need for one-hot
            y_encoded = y.reshape(-1, 1)

        # Split data into training and validation sets (10% for validation)
        val_size = int(0.1 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        X_train, y_train = X[train_indices], y_encoded[train_indices]
        X_val, y_val = X[val_indices], y_encoded[val_indices]

        # Initialize weights and bias
        if self.n_classes == 2:
            output_dim = 1
        else:
            output_dim = self.n_classes

        if self.W is None:
            self.W = np.random.randn(n_features, output_dim) * 0.01
        if self.b is None:
            self.b = np.zeros((1, output_dim))

        # Initialize variables for early stopping
        best_val_loss = float("inf")
        best_W = self.W.copy()
        best_b = self.b.copy()
        consecutive_increases = 0

        # Training loop
        for epoch in range(max_epochs):
            # Create batches
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            n_batches = int(np.ceil(len(X_train) / batch_size))
            epoch_loss = 0

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # Forward pass
                z = np.dot(X_batch, self.W) + self.b

                if self.n_classes == 2:
                    # Binary classification
                    y_pred = self.sigmoid(z)
                    # Binary cross-entropy loss
                    epsilon = 1e-15  # Prevent log(0)
                    loss = -np.mean(y_batch * np.log(y_pred + epsilon) + (1 - y_batch) * np.log(1 - y_pred + epsilon))

                    # Compute gradients
                    dz = y_pred - y_batch
                    dW = (1 / len(X_batch)) * np.dot(X_batch.T, dz)
                    db = (1 / len(X_batch)) * np.sum(dz, axis=0, keepdims=True)
                else:
                    # Multi-class classification
                    y_pred = self.softmax(z)
                    # Cross-entropy loss
                    epsilon = 1e-15  # Prevent log(0)
                    loss = -np.mean(np.sum(y_batch * np.log(y_pred + epsilon), axis=1))

                    # Compute gradients
                    dz = y_pred - y_batch
                    dW = (1 / len(X_batch)) * np.dot(X_batch.T, dz)
                    db = (1 / len(X_batch)) * np.sum(dz, axis=0, keepdims=True)

                # Add regularization if specified
                if regularization > 0:
                    loss += regularization * np.sum(self.W**2)
                    dW += 2 * regularization * self.W

                epoch_loss += loss * len(X_batch) / len(X_train)

                # Update parameters
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            # Evaluate on validation set
            z_val = np.dot(X_val, self.W) + self.b

            if self.n_classes == 2:
                y_val_pred = self.sigmoid(z_val)
                # Binary cross-entropy loss
                epsilon = 1e-15  # Prevent log(0)
                val_loss = -np.mean(
                    y_val * np.log(y_val_pred + epsilon) + (1 - y_val) * np.log(1 - y_val_pred + epsilon)
                )
            else:
                y_val_pred = self.softmax(z_val)
                # Cross-entropy loss
                epsilon = 1e-15  # Prevent log(0)
                val_loss = -np.mean(np.sum(y_val * np.log(y_val_pred + epsilon), axis=1))

            if regularization > 0:
                val_loss += regularization * np.sum(self.W**2)

            # Save losses for plotting
            self.train_losses.append(epoch_loss)
            self.val_losses.append(val_loss)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = self.W.copy()
                best_b = self.b.copy()
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Set model parameters to the best ones found during training
        self.W = best_W
        self.b = best_b

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for the given input data.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features)

        Returns:
        numpy.ndarray: Predicted probabilities of shape (n_samples, n_classes)
        """
        z = np.dot(X, self.W) + self.b

        if self.n_classes == 2:
            # Binary classification
            proba = self.sigmoid(z)
            return np.hstack([1 - proba, proba])  # Return probabilities for both classes
        else:
            # Multi-class classification
            return self.softmax(z)

    def predict(self, X):
        """
        Predicts class labels for the given input data.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features)

        Returns:
        numpy.ndarray: Predicted class labels of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """
        Computes the accuracy of the model.

        Parameters:
        X (numpy.ndarray): Input features of shape (n_samples, n_features)
        y (numpy.ndarray): True class labels of shape (n_samples,)

        Returns:
        float: Accuracy (proportion of correctly classified samples)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def save(self, filepath):
        """
        Saves the model parameters to a file.

        Parameters:
        filepath (str): Path to save the model parameters
        """
        model_params = {"W": self.W, "b": self.b, "classes_": self.classes_, "n_classes": self.n_classes}

        with open(filepath, "wb") as f:
            pickle.dump(model_params, f)

    def load(self, filepath):
        """
        Loads the model parameters from a file.

        Parameters:
        filepath (str): Path to the saved model parameters

        Returns:
        self: The model with loaded parameters
        """
        with open(filepath, "rb") as f:
            model_params = pickle.load(f)

        self.W = model_params["W"]
        self.b = model_params["b"]
        self.classes_ = model_params["classes_"]
        self.n_classes = model_params["n_classes"]

        return self
