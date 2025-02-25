"""This module trains a Linear Regression Model to predict Petal Width given Petal Length."""

import logging
import os

from lr.dataset import load_iris_data, plot_training_loss
from lr.linear_regression import LinearRegression

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to train the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    X_train_features = X_train[:, 2].reshape(-1, 1)  # Petal length  # noqa: N806
    y_train_target = X_train[:, 3]  # Petal width

    model = LinearRegression(learning_rate=0.01)
    model.fit(X_train_features, y_train_target, batch_size=32, max_epochs=100, patience=3)

    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/regression2.pkl")

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_training_loss(model, "Training and Validation Loss: Petal Length → Petal Width", "plots/regression2_loss.png")

    logger.info("Model 2 trained and saved successfully.")


if __name__ == "__main__":
    main()
