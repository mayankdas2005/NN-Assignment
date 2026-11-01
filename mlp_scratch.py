"""
Multilayer Perceptron implemented from scratch with NumPy only.

Supports any number of hidden layers (ReLU) and two kinds of output layer:
  * "softmax" + categorical cross-entropy  -> multi-class (Iris)
  * "sigmoid" + binary cross-entropy       -> binary (Breast Cancer)

Training uses plain (full-batch) gradient descent:  W <- W - lr * dL/dW
"""
import numpy as np


# ----------------------------------------------------------------------------
# Activation functions and their derivatives
# ----------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    z = np.clip(z, -500, 500)          # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------
EPS = 1e-12


def categorical_cross_entropy(Y_onehot, P):
    """L = -(1/m) * sum_i sum_k  y_ik * log(p_ik)"""
    P = np.clip(P, EPS, 1 - EPS)
    return -np.mean(np.sum(Y_onehot * np.log(P), axis=1))


def binary_cross_entropy(y, p):
    """L = -(1/m) * sum_i [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]"""
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


# ----------------------------------------------------------------------------
# The MLP
# ----------------------------------------------------------------------------
class MLP:
    """
    layer_sizes : e.g. [4, 6, 3]  -> 4 inputs, one hidden layer of 6, 3 outputs
                  e.g. [30, 15, 8, 1] -> two hidden layers (15, 8), 1 output
    output      : "softmax" (multi-class) or "sigmoid" (binary)
    """

    def __init__(self, layer_sizes, output="softmax", seed=42):
        if output not in ("softmax", "sigmoid"):
            raise ValueError("output must be 'softmax' or 'sigmoid'")
        self.layer_sizes = list(layer_sizes)
        self.output = output
        self.n_layers = len(layer_sizes) - 1          # number of weight matrices
        self.W, self.b = self._initialize(seed)
        # keep a copy of the initial parameters to show what learning changed
        self.W_init = [w.copy() for w in self.W]
        self.b_init = [b.copy() for b in self.b]
        self.history = {"train_loss": [], "test_loss": []}

    # ---------------------------------------------------------------- init --
    def _initialize(self, seed):
        """
        Hidden layers (followed by ReLU): He initialisation  W ~ N(0, 2/n_in)
        Output layer (softmax/sigmoid):   Xavier initialisation W ~ N(0, 1/n_in)
        Biases start at zero.
        """
        rng = np.random.default_rng(seed)
        W, b = [], []
        for l, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            is_output = (l == self.n_layers - 1)
            std = np.sqrt(1.0 / n_in) if is_output else np.sqrt(2.0 / n_in)
            W.append(rng.normal(0.0, std, size=(n_in, n_out)))
            b.append(np.zeros((1, n_out)))
        return W, b

    # ------------------------------------------------------------- forward --
    def forward(self, X):
        """
        A[0] = X
        Z[l] = A[l-1] @ W[l] + b[l]
        A[l] = ReLU(Z[l])                    for hidden layers
        A[L] = softmax(Z[L]) or sigmoid(Z[L]) for the output layer
        Returns the output and a cache of (Z, A) needed for backprop.
        """
        A = X
        cache = {"A": [X], "Z": []}
        for l in range(self.n_layers):
            Z = A @ self.W[l] + self.b[l]
            if l < self.n_layers - 1:
                A = relu(Z)
            else:
                A = softmax(Z) if self.output == "softmax" else sigmoid(Z)
            cache["Z"].append(Z)
            cache["A"].append(A)
        return A, cache

    # ---------------------------------------------------------------- loss --
    def compute_loss(self, Y, P):
        if self.output == "softmax":
            return categorical_cross_entropy(Y, P)
        return binary_cross_entropy(Y, P)

    # ------------------------------------------------------------ backward --
    def backward(self, Y, cache):
        """
        Output layer: for softmax+CCE and for sigmoid+BCE the gradient of the
        loss w.r.t. the pre-activation simplifies to the same expression:
              dZ[L] = (P - Y) / m
        Then for every layer l (going backwards):
              dW[l] = A[l-1]^T @ dZ[l]
              db[l] = sum over samples of dZ[l]
              dA[l-1] = dZ[l] @ W[l]^T
              dZ[l-1] = dA[l-1] * ReLU'(Z[l-1])
        """
        m = Y.shape[0]
        A, Z = cache["A"], cache["Z"]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        dZ = (A[-1] - Y) / m
        for l in reversed(range(self.n_layers)):
            dW[l] = A[l].T @ dZ
            db[l] = dZ.sum(axis=0, keepdims=True)
            if l > 0:
                dA_prev = dZ @ self.W[l].T
                dZ = dA_prev * relu_derivative(Z[l - 1])
        return dW, db

    # -------------------------------------------------------------- update --
    def update(self, dW, db, lr):
        """Gradient descent:  W <- W - lr * dW ,  b <- b - lr * db"""
        for l in range(self.n_layers):
            self.W[l] -= lr * dW[l]
            self.b[l] -= lr * db[l]

    # ----------------------------------------------------------------- fit --
    def _prepare_targets(self, y):
        y = np.asarray(y)
        if self.output == "softmax":
            return one_hot(y.astype(int), self.layer_sizes[-1])
        return y.reshape(-1, 1).astype(float)

    def fit(self, X, y, epochs=1000, lr=0.1, X_test=None, y_test=None, verbose=0):
        """Full-batch gradient descent. y holds integer class labels (0..K-1)."""
        Y = self._prepare_targets(y)
        Y_test = self._prepare_targets(y_test) if y_test is not None else None
        for epoch in range(1, epochs + 1):
            P, cache = self.forward(X)
            loss = self.compute_loss(Y, P)
            dW, db = self.backward(Y, cache)
            self.update(dW, db, lr)

            self.history["train_loss"].append(loss)
            if X_test is not None:
                P_test, _ = self.forward(X_test)
                self.history["test_loss"].append(self.compute_loss(Y_test, P_test))
            if verbose and (epoch == 1 or epoch % verbose == 0):
                print(f"epoch {epoch:5d} | train loss {loss:.4f}")
        return self

    # ------------------------------------------------------------- predict --
    def predict_proba(self, X):
        P, _ = self.forward(X)
        return P

    def predict(self, X, threshold=0.5):
        P = self.predict_proba(X)
        if self.output == "softmax":
            return P.argmax(axis=1)
        return (P.ravel() >= threshold).astype(int)

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def __repr__(self):
        return f"MLP({'-'.join(map(str, self.layer_sizes))}, output={self.output})"


# ----------------------------------------------------------------------------
# Gradient check: compares backprop gradients with numerical finite differences
# ----------------------------------------------------------------------------
def gradient_check(model, X, y, eps=1e-5):
    """Returns the relative error between analytical and numerical gradients
    for every weight matrix and bias vector. Values < 1e-6 mean backprop is correct."""
    Y = model._prepare_targets(y)
    _, cache = model.forward(X)
    dW, db = model.backward(Y, cache)

    def numerical(param):
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps
            loss_plus = model.compute_loss(Y, model.forward(X)[0])
            param[idx] = old - eps
            loss_minus = model.compute_loss(Y, model.forward(X)[0])
            param[idx] = old
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    results = []
    for l in range(model.n_layers):
        for name, analytic, param in (("W", dW[l], model.W[l]), ("b", db[l], model.b[l])):
            num = numerical(param)
            rel = np.linalg.norm(analytic - num) / (np.linalg.norm(analytic) + np.linalg.norm(num) + 1e-12)
            results.append((f"{name}{l + 1}", param.shape, rel))
    return results
