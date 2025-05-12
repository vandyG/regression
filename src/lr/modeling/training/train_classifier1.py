import os

from lr.dataset import (
    load_iris_data_for_classification,
    plot_classifier_decision_regions,
    plot_training_loss_logistic,
)
from lr.logistic_regression import LogisticRegression


def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_iris_data_for_classification()

    # Model 1: Use petal length/width (features 2 and 3)
    X_train_features = X_train[:, 2:4]  # Petal length and width

    # Create and train the model
    model = LogisticRegression(learning_rate=0.05)
    model.fit(X_train_features, y_train, batch_size=32, max_epochs=200, patience=10)

    # Create directories for models and plots if they don't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the model
    model.save("models/classifier1.pkl")

    # Plot and save the loss curves
    plot_training_loss_logistic(
        model,
        "Training and Validation Loss: Logistic Regression with Petal Features",
        "plots/classifier1_loss.png",
    )

    # Plot decision regions
    plot_classifier_decision_regions(
        model,
        X_train_features,
        y_train,
        "Decision Regions: Logistic Regression with Petal Features",
        "plots/classifier1_decision_regions.png",
        ["Petal Length", "Petal Width"],
    )

    # Evaluate on training set
    train_accuracy = model.score(X_train_features, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    print("Model 1 (Petal Features) trained and saved successfully.")


if __name__ == "__main__":
    main()
