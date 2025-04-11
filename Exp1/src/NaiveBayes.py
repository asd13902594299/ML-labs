import random
import sys
from utils import build_cross_validate, k_cross_validation, train_and_validate
from math import log
from abc import abstractmethod


class NaiveBayesBase:
    def __init__(self):
        self.total_map: dict = {}
        self.total_count: int = 0
        self.ham_map: dict = {}
        self.ham_count: int = 0
        self.spam_map: dict = {}
        self.spam_count: int = 0
        # use for laplace smoothing
        self.vocab_appear: set = set()
        self.vocab_count: int = 0

    def train(self, data: list[list[str]], label: list[int]):
        for i in range(len(data)):
            for word in data[i]:
                if label[i] == 0:
                    self.ham_map[word] = self.ham_map.get(word, 0) + 1
                    self.ham_count += 1
                else:
                    self.spam_map[word] = self.spam_map.get(word, 0) + 1
                    self.spam_count += 1
                self.total_map[word] = self.total_map.get(word, 0) + 1
                self.total_count += 1
                if word not in self.vocab_appear:
                    self.vocab_appear.add(word)
                    self.vocab_count += 1

    @abstractmethod
    def predict(self, data: list[str]) -> int:
        pass

    def clean(self):
        self.total_map.clear()
        self.total_count = 0
        self.ham_map.clear()
        self.ham_count = 0
        self.spam_map.clear()
        self.spam_count = 0
        self.vocab_appear.clear()
        self.vocab_count = 0


class NaiveBayes(NaiveBayesBase):
    def predict(self, data: list[str]) -> int:
        # using MAP normal multiplication to do the prediction
        ham_prob = self.ham_count / self.total_count
        spam_prob = self.spam_count / self.total_count
        for word in data:
            ham_prob *= self.ham_map.get(word, 0) / self.ham_count
            spam_prob *= self.spam_map.get(word, 0) / self.spam_count
        return 0 if ham_prob > spam_prob else 1


class SmoothNaiveBayes(NaiveBayesBase):
    def predict(self, data: list[str]) -> int:
        # using log addition to do the prediction
        ham_prob = log(self.ham_count / self.total_count)
        spam_prob = log(self.spam_count / self.total_count)
        for word in data:
            # using Laplace smoothing
            ham_prob += log((self.ham_map.get(word, 0) + 1) / (self.ham_count + self.vocab_count))
            spam_prob += log((self.spam_map.get(word, 0) + 1) / (self.spam_count + self.vocab_count))
        return 0 if ham_prob > spam_prob else 1


if __name__ == '__main__':
    seed: int = 100
    random.seed(seed)

    k: int = 5
    total_data: int = 37823
    cross_validation_path: str = "./cross_validation/"

    classifier = NaiveBayes()

    # Parse argument
    arg = sys.argv[1] if len(sys.argv) > 1 else "default"

    # for Q1 training with 5% and validating with 95%
    if arg == "Q1.1":
        build_cross_validate(cross_validation_path, 20, total_data)
        train_and_validate(classifier, cross_validation_path, 0, list(range(1, 20)))
    # for Q1 training with 50% and validating with 50%
    elif arg == "Q1.2":
        build_cross_validate(cross_validation_path, 2, total_data)
        train_and_validate(classifier, cross_validation_path, 0, [1])
    # for Q1 training with 100% and validating with 100%
    elif arg == "Q1.3":
        build_cross_validate(cross_validation_path, 1, total_data)
        train_and_validate(classifier, cross_validation_path, 0, [0])
    # default k-cross validation
    else:
        build_cross_validate(cross_validation_path, k, total_data)
        k_cross_validation(cross_validation_path, k, classifier)
