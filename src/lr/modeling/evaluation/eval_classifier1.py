from lr.logistic_regression import LogisticRegression
from lr.dataset import load_iris_data_for_classification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_iris_data_for_classification()

    # Prepare test data for Model 1: Use petal length/width (features 2 and 3)
    X_test_features = X_test[:, 2:4]  # Petal length and width

    # Load the model
    model = LogisticRegression()
    model.load("models/classifier1.pkl")

    # Evaluate the model
    y_pred = model.predict(X_test_features)
    accuracy = model.score(X_test_features, y_test)

    print(f"Logistic Regression with Petal Features")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
