import numpy as np


class LinearRegression:
    """
    A vanilla linear regression model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model for the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): the target

        Returns:
            None

        """
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.b = self.w[0]
        self.w = self.w[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.insert(X, 0, 1, axis=1)
        return X @ np.insert(self.w, 0, self.b)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model for the given input.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): the target
            lr (float): learning rate
            epochs (int): number of times of running the model

        Returns:
            None

        """
        # X = X.astype(np.float32)  # convert data type to float32
        # y = y.astype(np.float32)
        # X = (X - X.mean()) / X.std()
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features)
        self.b = np.random.random()
        for _ in range(epochs):
            y_pred = X @ self.w + self.b
            residuals = y_pred - y
            gradient_weights = (2 / n_samples) * (X.T @ residuals)
            gradient_bias = (2 / n_samples) * np.sum(residuals)
            self.w -= lr * gradient_weights
            self.b -= lr * gradient_bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w + self.b
