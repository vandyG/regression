"""This module evaluates a previously trained model to predict Petal Width given all other features.

This module tests two Linear Regression model with and without `Regularization`.
"""

from logging import getLogger

from lr.dataset import load_iris_data
from lr.linear_regression import LinearRegression

logger = getLogger(__name__)


def main() -> None:
    """Main function to test the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    X_test_features = X_test[:, 0:3]  # All features except petal width  # noqa: N806
    y_test_target = X_test[:, 3]  # Petal width

    model_no_reg = LinearRegression()
    model_no_reg.load("models/regression4_no_reg.pkl")

    model_reg = LinearRegression()
    model_reg.load("models/regression4_with_reg.pkl")

    mse_no_reg = model_no_reg.score(X_test_features, y_test_target)
    mse_reg = model_reg.score(X_test_features, y_test_target)

    print("Model 4 - All Features → Petal Width")
    print(f"Test Mean Squared Error (No Regularization): {mse_no_reg:.6f}")
    print(f"Test Mean Squared Error (With Regularization): {mse_reg:.6f}")
    print(f"Difference in MSE: {mse_no_reg - mse_reg:.6f}")


if __name__ == "__main__":
    main()
