"""Implementation of Linear Regression using Gradient Descent."""

import pickle
from logging import getLogger
from typing import Self

import numpy as np

logger = getLogger(__name__)


class LinearRegression:
    """Implementation of Linear Regression using Gradient Descent.

    This class implements a simple linear regression model that utilizes gradient descent
    for optimizing model parameters. It supports training with mini-batches, L2 regularization,
    and early stopping based on validation loss. Additionally, the model parameters can be saved
    to and loaded from disk for persistence.

    Attributes:
        learning_rate (float): Learning rate for gradient descent updates.
        W (numpy.ndarray): Weights of the model. Initialized during training.
        b (numpy.ndarray): Bias term of the model. Initialized during training.
        train_losses (list): History of training losses per epoch.
        val_losses (list): History of validation losses per epoch.

    Methods:
        fit(X, y, batch_size, regularization, max_epochs, patience):
            Trains the model using gradient descent on the provided data.
        predict(X):
            Predicts target values for the given input features.
        score(X, y):
            Computes the mean squared error between predictions and actual target values.
        save(filepath):
            Saves the model parameters to a file.
        load(filepath):
            Loads the model parameters from a file.
    """

    def __init__(self, learning_rate: float = 0.01):
        """Initializes the instance based on learning rate.

        Args:
            learning_rate (float, optional): Learning rate for gradient descent updates. Defaults to 0.01.
        """
        self.learning_rate = learning_rate
        self.W = None
        self.b = None
        self.train_losses = []
        self.val_losses = []

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        batch_size: int = 32,
        regularization: float = 0,
        max_epochs: int = 100,
        patience: int = 3,
    ) -> Self:
        """Trains the model using gradient descent on the given data.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples, n_outputs) or (n_samples,).
            batch_size (int, optional): Size of each batch during training. Defaults to 32.
            regularization (float, optional): L2 regularization factor. Defaults to 0.
            max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 3.

        Returns:
            LinearRegression: The trained model.
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        val_size = int(0.1 * n_samples)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        X_train, y_train = X[train_indices], y[train_indices]  # noqa: N806
        X_val, y_val = X[val_indices], y[val_indices]  # noqa: N806

        if self.W is None:
            self.W = np.random.randn(n_features, n_outputs) * 0.01
        if self.b is None:
            self.b = np.zeros((1, n_outputs))

        best_val_loss = float("inf")
        best_W = self.W.copy()  # noqa: N806
        best_b = self.b.copy()
        consecutive_increases = 0

        for epoch in range(max_epochs):
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]  # noqa: N806
            y_train = y_train[indices]

            n_batches = int(np.ceil(len(X_train) / batch_size))
            epoch_loss = 0

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(X_train))

                X_batch = X_train[start_idx:end_idx]  # noqa: N806
                y_batch = y_train[start_idx:end_idx]

                y_pred = np.dot(X_batch, self.W) + self.b

                batch_loss = np.mean((y_pred - y_batch) ** 2)
                if regularization > 0:
                    batch_loss += regularization * np.sum(self.W**2)

                epoch_loss += batch_loss * len(X_batch) / len(X_train)

                dW = (2 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))  # noqa: N806
                if regularization > 0:
                    dW += 2 * regularization * self.W  # noqa: N806

                db = (2 / len(X_batch)) * np.sum(y_pred - y_batch, axis=0, keepdims=True)

                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

            y_val_pred = self.predict(X_val)
            val_loss = np.mean((y_val_pred - y_val) ** 2)

            self.train_losses.append(epoch_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = self.W.copy()  # noqa: N806
                best_b = self.b.copy()
                consecutive_increases = 0
            else:
                consecutive_increases += 1
                if consecutive_increases >= patience:
                    logger.warning(f"Early stopping at epoch {epoch + 1}")
                    break

        self.W = best_W
        self.b = best_b

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Predicts target values for the given input data.

        Args:
            X (numpy.ndarray): Input features of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Predicted values of shape (n_samples, n_outputs)
        """
        return np.dot(X, self.W) + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # noqa: N803
        """Computes the mean squared error between the predicted values and the target values.

        Args:
            X (numpy.ndarray): Input features of shape (n_samples, n_features)
            y (numpy.ndarray): Target values of shape (n_samples, n_outputs) or (n_samples,)

        Returns:
            float: Mean squared error.
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)

    def save(self, filepath: str) -> None:
        """Saves the model parameters to a file.

        Args:
            filepath (str): Path to save the model parameters.
        """
        model_params = {
            "W": self.W,
            "b": self.b,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_params, f)

    def load(self, filepath: str) -> Self:
        """Loads the model parameters from a file.

        Args:
            filepath (str): Path to the saved model parameters.

        Returns:
            self: The model with loaded parameters.
        """
        with open(filepath, "rb") as f:
            model_params = pickle.load(f)  # noqa: S301

        self.W = model_params["W"]
        self.b = model_params["b"]

        return self
