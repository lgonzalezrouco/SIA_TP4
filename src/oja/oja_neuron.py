import numpy as np


class OjaNeuron:
    def __init__(
        self,
        n_features,
        learning_rate=1e-3,
        max_epochs=1000,
        tol=1e-6,
        seed=42,
        decay_type="constant",
    ):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(0, 1, size=n_features)
        self.lr_0 = learning_rate
        self.decay_type = decay_type
        self.max_epochs = max_epochs
        self.tol = tol
        self.history = []

    def fit(self, X):
        for epoch in range(self.max_epochs):
            # Calculate current learning rate
            if self.decay_type == "constant":
                lr = self.lr_0
            elif self.decay_type == "inverse":
                lr = self.lr_0 / (1 + epoch)
            elif self.decay_type == "exponential":
                lr = self.lr_0 * np.exp(-epoch / self.max_epochs)
            else:
                lr = self.lr_0

            w_old = self.w.copy()
            indices = np.random.permutation(len(X))

            for i in indices:
                x = X[i]
                y = np.inner(x, self.w)
                self.w += lr * y * (x - y * self.w)

            delta = np.linalg.norm(self.w - w_old)
            self.history.append(delta)

            if delta < self.tol:
                print(f"Oja converge en época {epoch + 1} con un delta de {delta:.2e}")
                break

        return self
