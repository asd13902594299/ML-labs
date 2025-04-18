from collections import Counter

import numpy as np
from sklearn import clone
from tqdm_joblib import tqdm_joblib

from tqdm import tqdm
from joblib import Parallel, delayed


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10, random_state=42, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []
        self.rng = np.random.default_rng(random_state)

    def _fit_one(self, X, y, indices):
        """
        Train one single model with base_estimator on the given indices.
        Each calling of this function assign a different random seed to the model.
        Return the trained model.
        """
        model = clone(self.base_estimator)
        if hasattr(model, "random_state"):
            model.set_params(random_state=self.rng.integers(0, 2**32 - 1))
        model.fit(X[indices], y[indices])
        return model

    def fit(self, X, y):
        """
        Train the Bagging classifier parallelly.
        """
        np.random.seed(self.random_state)
        y = np.array(y)
        n_samples = X.shape[0]

        indices_list = [np.random.choice(n_samples, n_samples, replace=True)
                        for _ in range(self.n_estimators)]

        with tqdm_joblib(desc="Training estimators", total=self.n_estimators):
            self.models = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_one)(X, y, indices)
                for indices in indices_list
            )

    def fit_sequential(self, X, y):
        """
        Train the Bagging classifier sequentially.
        """
        n_samples = X.shape[0]
        y = np.array(y)

        for _ in tqdm(range(self.n_estimators)):
            indices = np.random.choice(n_samples, n_samples, replace=True)

            X_sample = X[indices]
            y_sample = y[indices]

            # Train a base classifier
            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X_pred):
        """
        Predict the class labels for the input samples with majority voting.
        """
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X_pred))  # shape (n_samples,)

        # Convert to shape (n_estimators, n_samples)
        predictions = np.stack(predictions, axis=0)

        # Majority vote across classifiers for each sample
        final_preds = []
        for i in range(predictions.shape[1]):
            counter = Counter(predictions[:, i])
            final_preds.append(counter.most_common(1)[0][0])

        return np.array(final_preds)
