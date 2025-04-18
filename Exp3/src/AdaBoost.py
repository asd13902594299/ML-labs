import numpy as np
from sklearn import clone
from tqdm import tqdm


class AdaBoostClassifier:
    def __init__(self, base_estimator, n_estimators=50, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.classes_ = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Train the AdaBoost classifier with SAMME algorithm.
        """
        y = np.array(y)
        n_samples = X.shape[0]

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples

        for _ in tqdm(range(self.n_estimators)):
            # Clone and fit base model
            model = clone(self.base_estimator)
            if hasattr(model, "random_state"):
                model.set_params(random_state=self.rng.integers(0, 2**32 - 1))
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)

            incorrect = (y_pred != y).astype(int)

            # Weighted error
            err = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

            # Stop if error too high or zero
            if err >= 1 - 1e-10 or err == 0:
                break

            # SAMME alpha
            alpha = np.log((1 - err) / (err + 1e-10)) + np.log(n_classes - 1)

            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predict class labels for samples in X using the trained AdaBoost classifier.
        Vote is weighted by the alpha values of the classifiers.
        """
        # Initialize an array to store the votes for each sample across all classes
        class_votes = np.zeros((X.shape[0], len(self.classes_)))

        # Collect weighted votes for each classifier
        for alpha, model in zip(self.alphas, self.models):
            y_pred = model.predict(X)
            for i, cls in enumerate(self.classes_):
                # Update votes for each sample
                class_votes[:, i] += alpha * (y_pred == cls)

        # Choose the class with the maximum vote for each sample
        final_preds = np.argmax(class_votes, axis=1)

        # Return the predicted class labels
        return self.classes_[final_preds]
