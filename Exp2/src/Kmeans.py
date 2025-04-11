import os
import pickle
import numpy as np
from ClusterVisualizer import ClusterVisualizer
from utils import accuracy, load_mnist_data
from tqdm import tqdm


class KMeansMNISTClassifier:
    def __init__(self, n_clusters=10, max_iters=30, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.cluster_labels = None
        self.visualizer = None

    def initialize_centroids(self, images) -> np.ndarray:
        """
        Randomly select initial centroids from the images
        """
        np.random.seed(self.random_state)
        indices = np.random.choice(len(images), self.n_clusters, replace=False)
        return images[indices]

    def assign_clusters(self, images) -> np.ndarray:
        """
        Assign clusters to the images based on the closest centroid
        """
        num_images = images.shape[0]
        distances = np.empty((num_images, self.n_clusters))  # Pre-allocate memory
        # Compute distance one by one [but slow:(]
        # Otherwise error givens in WSL
        # distances = np.linalg.norm(images[:, np.newaxis] - self.centroids, axis=2)
        for i in range(num_images):
            distances[i] = np.linalg.norm(images[i] - self.centroids, axis=1)

        return np.argmin(distances, axis=1)

    def update_centroids(self, images, clusters_labels):
        """
        Update centroids based on the mean of the assigned images
        """
        new_centroids = np.array([images[clusters_labels == i].mean(axis=0) if len(images[clusters_labels == i]) > 0 else self.centroids[i]
                                  for i in range(self.n_clusters)])
        return new_centroids

    def train(self, mnist_data, model_path=None, load_from_path=False):
        """
        Train the KMeans classifier on the MNIST dataset
        - Initialize centroids
        - do until convergence or max_iters:
            - Assign clusters
            - Update centroids
        - Assign labels to clusters based on the most frequent label in each cluster
        """
        # load the trained model
        if load_from_path and model_path and os.path.exists(model_path):
            # Load the model from the saved file
            with open(model_path, 'rb') as f:
                saved_model = pickle.load(f)
                self.centroids = saved_model['centroids']
                self.cluster_labels = saved_model['cluster_labels']
            print(f"Model loaded from {model_path}")
            return

        labels, images = zip(*mnist_data)
        # image: (60000, 28, 28), labels: (60000, 1)
        images = np.array(images).reshape(len(images), -1)  # Flatten images
        labels = np.array(labels)
        # image: (60000, 784), labels: (60000, 1)

        self.centroids = self.initialize_centroids(images)
        for _ in tqdm(range(self.max_iters)):
            cluster_assignments = self.assign_clusters(images)
            new_centroids = self.update_centroids(images, cluster_assignments)

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:  # converge
                break
            self.centroids = new_centroids

        # map each cluster to the most frequent label
        self.cluster_labels = {}
        for i in range(self.n_clusters):
            cluster_members = labels[np.where(cluster_assignments == i)]
            # assign the most frequent label to that cluster
            self.cluster_labels[i] = np.bincount(cluster_members).argmax() if len(cluster_members) > 0 else -1

        # Save the trained model
        if model_path:
            model_data = {
                'centroids': self.centroids,
                'cluster_labels': self.cluster_labels
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {model_path}")

    def predict(self, image):
        """
        predict the given image to the label
        """
        image = image.flatten().reshape(1, -1)  # Flatten input image
        distances = np.linalg.norm(image - self.centroids, axis=1)
        cluster = np.argmin(distances)
        return self.cluster_labels.get(cluster, -1)  # Return assigned digit

    def visualize_clusters(self, data, n_components, alg):
        self.visualizer = ClusterVisualizer(
            self.centroids, self.cluster_labels, self.n_clusters, self.random_state)
        if alg == "PCA":
            if n_components == 2:
                self.visualizer.visualize_clusters_pca_2d(data)
            elif n_components == 3:
                self.visualizer.visualize_clusters_pca_3d(data)
            else:
                raise ValueError("n_components must be 2 or 3 for PCA.")
        elif alg == "TSNE":
            if n_components == 2:
                self.visualizer.visualize_clusters_tsne_2d(data)
            elif n_components == 3:
                self.visualizer.visualize_clusters_tsne_3d(data)
            else:
                raise ValueError("n_components must be 2 or 3 for TSNE.")
        else:
            raise ValueError("Unsupported algorithm. Use 'PCA' or 'TSNE'.")


if __name__ == '__main__':
    label_path: str = "data/MNIST/raw/train-labels-idx1-ubyte"
    image_path: str = "data/MNIST/raw/train-images-idx3-ubyte"
    model_path: str = "models/kmeans_cluster64.pkl"
    # 60000 x 28 x 28
    mnist_data: list[tuple[int, np.ndarray]] = load_mnist_data(image_path, label_path)
    mnist_data = mnist_data[:10000]
    classifier = KMeansMNISTClassifier(n_clusters=10)
    classifier.train(mnist_data, model_path=None, load_from_path=False)
    print(accuracy(classifier, mnist_data))
