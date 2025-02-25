"""This module evaluates a previously trained model to predict Petal Width given Petal Length."""

from logging import getLogger

from lr.dataset import load_iris_data
from lr.linear_regression import LinearRegression

logger = getLogger(__name__)


def main() -> None:
    """Main function to test the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    X_test_features = X_test[:, 2].reshape(-1, 1)  # Petal length  # noqa: N806
    y_test_target = X_test[:, 3]  # Petal width

    model = LinearRegression()
    model.load("models/regression2.pkl")

    mse = model.score(X_test_features, y_test_target)
    logger.info("Model 2 - Petal Length → Petal Width")
    logger.info(f"Test Mean Squared Error: {mse:.6f}")


if __name__ == "__main__":
    main()
