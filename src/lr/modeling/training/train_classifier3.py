from lr.logistic_regression import LogisticRegression
from lr.dataset import load_iris_data_for_classification, plot_training_loss
import numpy as np
import os


def main():
    # Load the data
    X_train, X_test, y_train, y_test = load_iris_data_for_classification()

    # Model 3: Use all features
    X_train_features = X_train  # All features

    # Create and train the model
    model = LogisticRegression(learning_rate=0.05)
    model.fit(X_train_features, y_train, batch_size=32, max_epochs=200, patience=10)

    # Create directories for models and plots if they don't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the model
    model.save("models/classifier3.pkl")

    # Plot and save the loss curves
    plot_training_loss(
        model, "Training and Validation Loss: Logistic Regression with All Features", "plots/classifier3_loss.png"
    )

    # Note: We can't plot decision regions for 4D data directly
    # But we can evaluate the model

    # Evaluate on training set
    train_accuracy = model.score(X_train_features, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    print("Model 3 (All Features) trained and saved successfully.")


if __name__ == "__main__":
    main()
