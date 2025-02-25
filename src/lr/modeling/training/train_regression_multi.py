"""This module trains a Linear Regression Model to predict Petal Features given Sepal Features."""

import logging
import os

from lr.dataset import load_iris_data, plot_training_loss
from lr.linear_regression import LinearRegression

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to train the model."""
    # Load the data
    X_train, X_test = load_iris_data()  # noqa: N806

    # Multi-output regression: Predict petal length and width from sepal length and width
    X_train_features = X_train[:, 0:2]  # Sepal length and width  # noqa: N806
    y_train_target = X_train[:, 2:4]  # Petal length and width

    # Create and train the model
    model = LinearRegression(learning_rate=0.01)
    model.fit(X_train_features, y_train_target, batch_size=32, max_epochs=100, patience=3)

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/regression_multi.pkl")

    # Plot and save the loss curves
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_training_loss(
        model,
        "Training and Validation Loss: Sepal Features → Petal Features",
        "plots/regression_multi_loss.png",
    )

    logger.info("Multi-output regression model trained and saved successfully.")


if __name__ == "__main__":
    main()
