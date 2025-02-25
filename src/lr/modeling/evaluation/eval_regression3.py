"""This module evaluates a previously trained model to predict Petal Length given Sepal Features."""

from logging import getLogger

from lr.dataset import load_iris_data
from lr.linear_regression import LinearRegression

logger = getLogger(__name__)


def main() -> None:
    """Main function to test the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    # Prepare test data for Model 3: Predict petal length (feature 2) from sepal features (features 0 and 1)
    X_test_features = X_test[:, 0:2]  # Sepal length and width  # noqa: N806
    y_test_target = X_test[:, 2]  # Petal length

    # Load the model
    model = LinearRegression()
    model.load("models/regression3.pkl")

    # Evaluate the model
    mse = model.score(X_test_features, y_test_target)
    logger.info("Model 3 - Sepal Features → Petal Length")
    logger.info(f"Test Mean Squared Error: {mse:.6f}")


if __name__ == "__main__":
    main()
