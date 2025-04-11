import random
import sys
from NaiveBayes import NaiveBayes, SmoothNaiveBayes
from utils import build_cross_validate, train_and_validate, k_cross_validation

if __name__ == '__main__':
    seed: int = 100
    random.seed(seed)

    k: int = 5
    total_data: int = 37823
    cross_validation_path: str = "./cross_validation/"

    # Parse argument
    arg = sys.argv[1] if len(sys.argv) > 1 else "default"

    # for Q1 training with 5% and validating with 95%
    if arg == "Q1.1":
        classifier = SmoothNaiveBayes()
        build_cross_validate(cross_validation_path, 20, total_data)
        train_and_validate(classifier, cross_validation_path, 0, list(range(1, 20)))
    # for Q1 training with 50% and validating with 50%
    elif arg == "Q1.2":
        classifier = SmoothNaiveBayes()
        build_cross_validate(cross_validation_path, 2, total_data)
        train_and_validate(classifier, cross_validation_path, 0, [1])
    # for Q1 training with 100% and validating with 100%
    elif arg == "Q1.3":
        classifier = SmoothNaiveBayes()
        build_cross_validate(cross_validation_path, 1, total_data)
        train_and_validate(classifier, cross_validation_path, 0, [0])
    # for Q2 using smoothed prediction in NaiveBayes
    elif arg == "Q2":
        classifier = SmoothNaiveBayes()
        build_cross_validate(cross_validation_path, k, total_data)
        k_cross_validation(cross_validation_path, k, classifier)
    # for Q3 using header information in NaiveBayes
    elif arg == "Q3":
        classifier = SmoothNaiveBayes()
        build_cross_validate(cross_validation_path, k, total_data, header=True)
        k_cross_validation(cross_validation_path, k, classifier)
    # default k-cross validation using NaiveBayes without smoothing
    else:
        classifier = NaiveBayes()
        build_cross_validate(cross_validation_path, k, total_data)
        k_cross_validation(cross_validation_path, k, classifier)
