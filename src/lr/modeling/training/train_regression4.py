"""This module trains a Linear Regression Model to predict Petal Width given all other features.

This module fits two Linear Regression model with and without `Regularization`.
"""

import logging
import os

from lr.dataset import load_iris_data, plot_training_loss
from lr.linear_regression import LinearRegression

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to train the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    X_train_features = X_train[:, 0:3]  # All features except petal width  # noqa: N806
    y_train_target = X_train[:, 3]  # Petal width

    # Create and train the model without regularization first
    model_no_reg = LinearRegression(learning_rate=0.01)
    model_no_reg.fit(X_train_features, y_train_target, batch_size=32, regularization=0, max_epochs=100, patience=3)

    if not os.path.exists("models"):
        os.makedirs("models")
    model_no_reg.save("models/regression4_no_reg.pkl")

    # Now train with regularization
    model_reg = LinearRegression(learning_rate=0.01)
    model_reg.fit(X_train_features, y_train_target, batch_size=32, regularization=0.1, max_epochs=100, patience=3)

    model_reg.save("models/regression4_with_reg.pkl")

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_training_loss(
        model_no_reg,
        "Training and Validation Loss: All Features → Petal Width (No Regularization)",
        "plots/regression4_no_reg_loss.png",
    )

    plot_training_loss(
        model_reg,
        "Training and Validation Loss: All Features → Petal Width (With Regularization)",
        "plots/regression4_with_reg_loss.png",
    )

    # Print weights to compare regularized vs non-regularized models
    logger.info("Weights without regularization:")
    logger.info(model_no_reg.W)
    logger.info("\nWeights with regularization:")
    logger.info(model_reg.W)
    logger.info("\nDifference in weights:")
    logger.info(model_no_reg.W - model_reg.W)

    logger.info("Model 4 (with and without regularization) trained and saved successfully.")


if __name__ == "__main__":
    main()
