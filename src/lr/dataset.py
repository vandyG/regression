"""This is a helper module to load the iris dataset and plot the results of training."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from lr.linear_regression import LinearRegression


def load_iris_data() -> tuple:
    """Load the Iris dataset and split it into training and testing sets.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where X contains the features
            and y contains the target values
    """
    iris = load_iris()
    X = iris.data  # noqa: N806

    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42, stratify=iris.target)  # noqa: N806

    return X_train, X_test


def plot_training_loss(
    model: LinearRegression,
    title: str,
    filename: str,
) -> None:
    """Plot the training and validation loss curves.

    Args:
        model (LinearRegression): Trained model with stored loss values.
        title (str): Title for the plot.
        filename (str): Filename to save the plot.
    """
    plt.figure(figsize=(10, 6))
    steps = range(1, len(model.train_losses) + 1)
    plt.plot(steps, model.train_losses, label="Training Loss")
    plt.plot(steps, model.val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid(visible=True)
    plt.savefig(filename)
    plt.close()
