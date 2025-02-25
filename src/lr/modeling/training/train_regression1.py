"""This module trains a Linear Regression Model to predict Sepal Width given Sepal Length."""

import logging
import os

from lr.dataset import load_iris_data, plot_training_loss
from lr.linear_regression import LinearRegression

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to train the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    # Model 1: Predict sepal width (feature 1) from sepal length (feature 0)
    X_train_features = X_train[:, 0].reshape(-1, 1)  # Sepal length  # noqa: N806
    y_train_target = X_train[:, 1]  # Sepal width

    # Create and train the model
    model = LinearRegression(learning_rate=0.01)
    model.fit(X_train_features, y_train_target, batch_size=32, max_epochs=100, patience=3)

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/regression1.pkl")

    # Plot and save the loss curves
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_training_loss(model, "Training and Validation Loss: Sepal Length → Sepal Width", "plots/regression1_loss.png")

    logger.info("Model 1 trained and saved successfully.")


if __name__ == "__main__":
    main()
