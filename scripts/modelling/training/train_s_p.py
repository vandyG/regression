from lr.lr.lr import LinearRegression
from lr.dataset import load_iris_data, plot_training_loss
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    # Load the data
    X_train, X_test = load_iris_data()

    # Multi-output regression: Predict petal length and width from sepal length and width
    X_train_features = X_train[:, 0:2]  # Sepal length and width
    y_train_target = X_train[:, 2:4]  # Petal length and width

    # Create and train the model
    model = LinearRegression(learning_rate=0.01)
    model.fit(X_train_features, y_train_target, batch_size=32, max_epochs=100, patience=3)

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/regression_multi.pkl")

    # Plot and save the loss curves
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_training_loss(
        model, "Training and Validation Loss: Sepal Features → Petal Features", "plots/regression_multi_loss.png"
    )

    print("Multi-output regression model trained and saved successfully.")


if __name__ == "__main__":
    main()
