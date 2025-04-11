import numpy as np
from utils import accuracy, load_mnist_data
from Kmeans import KMeansMNISTClassifier

if __name__ == "__main__":
    args: dict = {
        "n_cluster": 128,
        "load_from_path": False,
        "visualize_dim": 2,  # 2 or 3
        "visualize_alg": "TSNE"  # PCA or TSNE
    }

    train_label_path: str = "data/MNIST/raw/train-labels-idx1-ubyte"
    train_image_path: str = "data/MNIST/raw/train-images-idx3-ubyte"
    test_label_path: str = "data/MNIST/raw/t10k-labels-idx1-ubyte"
    test_image_path: str = "data/MNIST/raw/t10k-images-idx3-ubyte"
    n_cluster: int = args["n_cluster"]
    load_from_path = args["load_from_path"]
    model_path: str = f"models/kmeans_cluster{n_cluster}.pkl"

    # 60000 x [(1), (28 x 28)]
    mnist_train: list[tuple[int, np.ndarray]] = load_mnist_data(train_image_path, train_label_path)
    classifier = KMeansMNISTClassifier(n_clusters=n_cluster)
    classifier.train(mnist_train, model_path=model_path, load_from_path=load_from_path)

    mnist_test: list[tuple[int, np.ndarray]] = load_mnist_data(test_image_path, test_label_path)
    classifier.visualize_clusters(mnist_test, n_components=args["visualize_dim"], alg=args["visualize_alg"])
    print(f'train_acc: {accuracy(classifier, mnist_train)}')
    print(f'test_acc: {accuracy(classifier, mnist_test)}')
