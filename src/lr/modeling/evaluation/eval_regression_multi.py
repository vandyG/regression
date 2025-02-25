"""This module evaluates a previously trained model to predict Petal Features given Sepal Features."""

from logging import getLogger

from lr.dataset import load_iris_data
from lr.linear_regression import LinearRegression

logger = getLogger(__name__)


def main() -> None:
    """Main function to test the model."""
    X_train, X_test = load_iris_data()  # noqa: N806

    X_test_features = X_test[:, 0:2]  # Sepal length and width  # noqa: N806
    y_test_target = X_test[:, 2:4]  # Petal length and width

    model = LinearRegression()
    model.load("models/regression_multi.pkl")

    mse = model.score(X_test_features, y_test_target)
    logger.info("Multi-Output Regression Model - Sepal Features → Petal Features")
    logger.info(f"Test Mean Squared Error: {mse:.6f}")

    predictions = model.predict(X_test_features)

    logger.info("\nSample predictions vs actual values:")
    logger.info("Predicted Petal Length, Predicted Petal Width | Actual Petal Length, Actual Petal Width")
    for i in range(min(5, len(predictions))):
        logger.info(
            f"{predictions[i][0]:.2f}, {predictions[i][1]:.2f} | {y_test_target[i][0]:.2f}, {y_test_target[i][1]:.2f}",
        )


if __name__ == "__main__":
    main()
