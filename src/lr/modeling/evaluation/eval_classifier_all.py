from lr.logistic_regression import LogisticRegression
from lr.dataset import load_iris_data_for_classification
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_iris_data_for_classification()

    # Load models
    model1 = LogisticRegression()
    model1.load("models/classifier1.pkl")

    model2 = LogisticRegression()
    model2.load("models/classifier2.pkl")

    model3 = LogisticRegression()
    model3.load("models/classifier3.pkl")

    # Evaluate models
    accuracy1 = model1.score(X_test[:, 2:4], y_test)  # Petal features
    accuracy2 = model2.score(X_test[:, 0:2], y_test)  # Sepal features
    accuracy3 = model3.score(X_test, y_test)  # All features

    print("Comparison of Logistic Regression Models:")
    print(f"Model 1 (Petal Features): Test Accuracy = {accuracy1:.4f}")
    print(f"Model 2 (Sepal Features): Test Accuracy = {accuracy2:.4f}")
    print(f"Model 3 (All Features): Test Accuracy = {accuracy3:.4f}")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    models = ["Petal Features", "Sepal Features", "All Features"]
    accuracies = [accuracy1, accuracy2, accuracy3]

    plt.bar(models, accuracies, color=["blue", "green", "red"])
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title("Comparison of Logistic Regression Models")
    plt.ylim([0, 1])

    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("plots/classifier_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
