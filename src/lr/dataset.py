"""This is a helper module to load the iris dataset and plot the results of training."""

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
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


def load_iris_data_for_classification():
    """Load the Iris dataset and split it into training and testing sets.

    Returns:
    tuple: (X_train, X_test, y_train, y_test) where X contains the features
           and y contains the target class labels
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def plot_training_loss_logistic(model, title, filename):
    """Plot the training and validation loss curves.

    Parameters:
    model (LogisticRegression): Trained model with stored loss values
    title (str): Title for the plot
    filename (str): Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    steps = range(1, len(model.train_losses) + 1)
    plt.plot(steps, model.train_losses, label="Training Loss")
    plt.plot(steps, model.val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_classifier_decision_regions(model, X, y, title, filename, feature_names):
    """Plot the decision regions for a classifier.

    Parameters:
    model (LogisticRegression): Trained classifier
    X (numpy.ndarray): Training data
    y (numpy.ndarray): Target values
    title (str): Title for the plot
    filename (str): Filename to save the plot
    feature_names (list): Names of the features used
    """
    plt.figure(figsize=(10, 8))

    # Create a copy of the model without the plotting method to avoid a potential issue with mlxtend
    class ModelWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            return self.model.predict(X)

    wrapper_model = ModelWrapper(model)

    # Plot decision regions
    plot_decision_regions(X, y, clf=wrapper_model, legend=2)

    # Add axis labels and title
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)

    # Add legend with class names
    plt.legend(["Setosa", "Versicolor", "Virginica"], loc="upper left")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
