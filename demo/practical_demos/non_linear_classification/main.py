"""
Non-Linear Classification with NeuraCommand

This tutorial demonstrates how to classify non-linearly separable datasets
using neural networks. We use the make_moons dataset from sklearn, which
creates two interleaving half circles that cannot be separated by a linear
decision boundary.

A multi-layer neural network with non-linear activation functions can learn
the complex decision boundary needed to classify such data.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neura_command.network.sequential import Sequential
from neura_command.layers import Dense, ReLU, Sigmoid
from neura_command.network_utils.optimizers import Adam
from neura_command.network_utils.loss_functions import MSELoss


def prepare_data(
    n_samples: int = 500, noise: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Generate and prepare the make_moons dataset.

    Args:
        n_samples: Number of samples to generate.
        noise: Standard deviation of Gaussian noise added to the data.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    # Generate the moons dataset
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # Reshape y for neural network (needs to be 2D)
    y = y.reshape(-1, 1).astype(np.float32)
    X = X.astype(np.float32)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Standardize features for better training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def build_model(input_size: int = 2, learning_rate: float = 0.01) -> Sequential:
    """
    Build a neural network for binary classification.

    The architecture uses:
    - Input layer: 2 features (x, y coordinates)
    - Hidden layer 1: 16 neurons with ReLU activation
    - Hidden layer 2: 8 neurons with ReLU activation
    - Output layer: 1 neuron with Sigmoid activation (binary classification)

    Args:
        input_size: Number of input features.
        learning_rate: Learning rate for the optimizer.

    Returns:
        Configured Sequential model.
    """
    model = Sequential(
        layers=[
            Dense(input_size, 16),
            ReLU(),
            Dense(16, 8),
            ReLU(),
            Dense(8, 1),
            Sigmoid(),
        ],
        optimizer=Adam(learning_rate=learning_rate),
        loss=MSELoss(),
    )
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
) -> dict:
    """
    Train the neural network model.

    Args:
        model: The Sequential model to train.
        X_train: Training features.
        y_train: Training labels.
        X_test: Validation features.
        y_test: Validation labels.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.

    Returns:
        Training history dictionary.
    """
    history = model.train(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=True,
    )
    return history


def evaluate_model(
    model: Sequential, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Evaluate the trained model.

    Args:
        model: Trained Sequential model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (accuracy, predictions).
    """
    # Make predictions
    predictions_prob = model.predict(X_test)
    predictions = (predictions_prob > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Class 0", "Class 1"]))

    return accuracy, predictions


def plot_decision_boundary(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
) -> None:
    """
    Plot the decision boundary of the trained model.

    Args:
        model: Trained Sequential model.
        X: Feature data (already scaled).
        y: Labels.
        title: Plot title.
    """
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Get predictions for the mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8, levels=50)
    plt.colorbar(label="Prediction Probability")

    # Plot decision boundary line
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    # Plot data points
    scatter = plt.scatter(
        X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdBu, edgecolors="black", s=50
    )

    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict) -> None:
    """
    Plot the training and validation loss over epochs.

    Args:
        history: Training history dictionary.
    """
    plt.figure(figsize=(10, 5))

    # Plot training loss
    plt.plot(history["loss"], label="Training Loss", color="blue")

    # Plot validation loss if available
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss", color="orange")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_dataset(X: np.ndarray, y: np.ndarray, title: str = "Moons Dataset") -> None:
    """
    Visualize the raw dataset.

    Args:
        X: Feature data.
        y: Labels.
        title: Plot title.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdBu, edgecolors="black", s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the non-linear classification tutorial.
    """
    print("=" * 60)
    print("Non-Linear Classification with NeuraCommand")
    print("Using the make_moons dataset")
    print("=" * 60)

    # Step 1: Prepare the data
    print("\n[Step 1] Preparing the moons dataset...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        n_samples=500, noise=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features shape: {X_train.shape}")

    # Visualize the dataset
    print("\n[Step 2] Visualizing the dataset...")
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.vstack([y_train, y_test])
    visualize_dataset(X_combined, y_combined, "Moons Dataset (Standardized)")

    # Step 3: Build the model
    print("\n[Step 3] Building the neural network model...")
    model = build_model(input_size=2, learning_rate=0.01)
    print(model.summary())

    # Step 4: Train the model
    print("\n[Step 4] Training the model...")
    history = train_model(
        model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32
    )

    # Step 5: Plot training history
    print("\n[Step 5] Plotting training history...")
    plot_training_history(history)

    # Step 6: Evaluate the model
    print("\n[Step 6] Evaluating the model...")
    accuracy, predictions = evaluate_model(model, X_test, y_test)

    # Step 7: Plot decision boundary
    print("\n[Step 7] Plotting decision boundary...")
    plot_decision_boundary(
        model,
        X_combined,
        y_combined,
        f"Non-Linear Decision Boundary (Accuracy: {accuracy * 100:.1f}%)",
    )

    print("\n" + "=" * 60)
    print("Tutorial completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
