from lr.lr.lr import LinearRegression
from lr.dataset import load_iris_data, plot_training_loss
import numpy as np
import os


def main():
    # Load the data
    X_train, X_test = load_iris_data()

    # Model 3: Predict petal length (feature 2) from sepal features (features 0 and 1)
    X_train_features = X_train[:, 0:2]  # Sepal length and width
    y_train_target = X_train[:, 2]  # Petal length

    # Create and train the model
    model = LinearRegression(learning_rate=0.01)
    model.fit(X_train_features, y_train_target, batch_size=32, max_epochs=100, patience=3)

    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/regression3.pkl")

    # Plot and save the loss curves
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plot_training_loss(
        model, "Training and Validation Loss: Sepal Features → Petal Length", "plots/regression3_loss.png"
    )

    print("Model 3 trained and saved successfully.")


if __name__ == "__main__":
    main()
