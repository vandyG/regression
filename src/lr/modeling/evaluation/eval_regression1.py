"""This module evaluates a previously trained model to predict Sepal Width given Sepal Length."""

from logging import getLogger

from lr.dataset import load_iris_data
from lr.linear_regression import LinearRegression

logger = getLogger(__name__)


def main() -> None:
    """Main function to test the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    # Prepare test data for Model 1: Predict sepal width (feature 1) from sepal length (feature 0)
    X_test_features = X_test[:, 0].reshape(-1, 1)  # Sepal length  # noqa: N806
    y_test_target = X_test[:, 1]  # Sepal width

    # Load the model
    model = LinearRegression()
    model.load("models/regression1.pkl")

    # Evaluate the model
    mse = model.score(X_test_features, y_test_target)
    print("Model 1 - Sepal Length → Sepal Width")
    print(f"Test Mean Squared Error: {mse:.6f}")


if __name__ == "__main__":
    main()
