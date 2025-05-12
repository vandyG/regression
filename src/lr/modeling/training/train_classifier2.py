from lr.logistic_regression import LogisticRegression
from lr.dataset import (
    load_iris_data_for_classification,
    plot_training_loss_logistic,
    plot_classifier_decision_regions,
)
import numpy as np
import os


def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_iris_data_for_classification()

    # Model 2: Use sepal length/width (features 0 and 1)
    X_train_features = X_train[:, 0:2]  # Sepal length and width

    # Create and train the model
    model = LogisticRegression(learning_rate=0.05)
    model.fit(X_train_features, y_train, batch_size=32, max_epochs=200, patience=10)

    # Create directories for models and plots if they don't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the model
    model.save("models/classifier2.pkl")

    # Plot and save the loss curves
    plot_training_loss_logistic(
        model, "Training and Validation Loss: Logistic Regression with Sepal Features", "plots/classifier2_loss.png"
    )

    # Plot decision regions
    plot_classifier_decision_regions(
        model,
        X_train_features,
        y_train,
        "Decision Regions: Logistic Regression with Sepal Features",
        "plots/classifier2_decision_regions.png",
        ["Sepal Length", "Sepal Width"],
    )

    # Evaluate on training set
    train_accuracy = model.score(X_train_features, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    print("Model 2 (Sepal Features) trained and saved successfully.")


if __name__ == "__main__":
    main()
