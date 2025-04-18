import csv
from sklearn.metrics import accuracy_score
import numpy as np


def get_data(csv_path: str) -> tuple[list[str], list[str]]:
    """
    Return the texts and ratings from the csv file.
    """
    texts = []
    ratings = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')  # or delimiter=',' if comma-separated
        for row in reader:
            texts.append(row['summary']+row['reviewText'])
            ratings.append(row['overall'])

    return texts, ratings


def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_model(y_true, y_pred):
    """
    Prints the accuracy, MAE and RMSE of the predictions.

    Parameters:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels
    """
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print()


if __name__ == "__main__":
    path: str = "data/exp3-reviews.csv"
    text, rating = get_data(path)
    for t, r in list(zip(text, rating))[0:5]:
        print(t, r)
        # print(f"Overall: {item[0]}, Summary: {item[1]}, reviewText: {item[2]}")
