import argparse
import os

from sklearn.tree import DecisionTreeClassifier
from utils import evaluate_model, get_data

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from AdaBoost import AdaBoostClassifier
from Bagging import BaggingClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_base_learners', type=int, help='Number of estimators', default=None)
    parser.add_argument('--base_learner', type=str, help='Base learner', default=None)
    parser.add_argument('--ensemble_method', type=str, help='Ensemble method', default='none')
    parser.add_argument('--dt_depth', type=int, help='Decision tree max depth', default=3)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--num_features', type=int, help='Number of features for vectorization', default=5000)
    parser.add_argument('--data_ratio', type=float, help='Ratio of data to used', default=1)
    parser.add_argument('--n_jobs', type=int, help='Number of jobs for parallel processing', default=-1)
    parser.add_argument('--load_model', type=bool, help='Enable loading model from ./models', default=True)
    parser.add_argument('--save_model', type=bool, help='Enable saving model to ./models', default=True)

    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)

    base_learner = args.base_learner
    ensemble_method = args.ensemble_method
    num_base_learners = args.num_base_learners
    seed = args.seed
    num_features = args.num_features
    data_ratio = args.data_ratio
    n_jobs = args.n_jobs
    load_model = args.load_model
    save_model = args.save_model
    dt_depth = args.dt_depth
    print(f"Base learner: {base_learner}")
    print(f"Ensemble method: {ensemble_method}")

    # Load data
    texts, ratings = get_data("data/exp3-reviews.csv")
    texts = texts[:int(len(texts) * data_ratio)]
    ratings = ratings[:int(len(ratings) * data_ratio)]
    X_train, X_test, y_train, y_test = train_test_split(texts, ratings, test_size=0.1, random_state=seed)

    # Vectorization for text data
    vectorizer_path = f"./models/vectorizer_{num_features}.pkl"
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
    else:
        vectorizer = TfidfVectorizer(max_features=num_features)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        joblib.dump(vectorizer, vectorizer_path)

    # Initialize the base classifier and ensemble method
    if ensemble_method == 'none':
        if base_learner == 'dt':
            print(f"Decision tree depth: {dt_depth}")
            model_path = f"./models/none_dt{dt_depth}_model.pkl"
            base_estimator = DecisionTreeClassifier(max_depth=dt_depth, random_state=seed)
        elif base_learner == 'nb':
            model_path = "./models/none_nb_model.pkl"
            base_estimator = MultinomialNB()
        else:
            raise ValueError("Unknown base learner. Use 'dt' or 'nb'.")
        ensemble_estimator = base_estimator

    elif ensemble_method in ['bagging', 'adaboost']:
        print(f"Number of estimators: {num_base_learners}")
        if base_learner == 'dt':
            print(f"Decision tree depth: {dt_depth}")
            model_path = f"./models/{ensemble_method}_dt{dt_depth}_{num_base_learners}_model.pkl"
            # No seed here to allow different learners
            base_estimator = DecisionTreeClassifier(max_depth=dt_depth)
        elif base_learner == 'nb':
            model_path = f"./models/{ensemble_method}_nb_{num_base_learners}_model.pkl"
            base_estimator = MultinomialNB()
        else:
            raise ValueError("Unknown base learner. Use 'dt' or 'nb'.")

        if ensemble_method == 'adaboost':
            ensemble_estimator = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=num_base_learners,
                random_state=seed
            )
        else:  # ensemble_method == 'bg'
            ensemble_estimator = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=num_base_learners,
                n_jobs=n_jobs,
                random_state=seed
            )
    else:
        raise ValueError("Unknown ensemble method. Use 'none', 'bagging', or 'adaboost'.")

    # Train/Save or load the ensemble_estimator
    if load_model and os.path.exists(model_path):
        print("Loading saved model...")
        ensemble_estimator = joblib.load(model_path)
    else:
        print("Training the model...")
        ensemble_estimator.fit(X_train, y_train)
        if save_model:
            print("Saving the model...")
            joblib.dump(ensemble_estimator, model_path)

    # Make predictions
    y_pred = ensemble_estimator.predict(X_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred)
