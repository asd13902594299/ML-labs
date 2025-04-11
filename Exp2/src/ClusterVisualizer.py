import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ClusterVisualizer:
    def __init__(self, centroids, cluster_labels, n_cluster, random_state=42):
        self.centroids = centroids
        self.cluster_labels = cluster_labels
        self.random_state = random_state
        self.n_cluster = n_cluster

    def _prepare_data(self, mnist_data, dim_reduction, n_components):
        labels, images = zip(*mnist_data)
        images = np.array(images).reshape(len(images), -1)  # Flatten images
        all_data = np.vstack([images, self.centroids])
        reduced_all = dim_reduction(n_components=n_components,
                                    random_state=self.random_state).fit_transform(all_data)
        return labels, reduced_all[:-len(self.centroids)], reduced_all[-len(self.centroids):]

    def _plot_2d(self, reduced_images, reduced_centroids, labels, title, xlabel, ylabel):
        unique_labels = np.unique(labels)
        color_map = {label: plt.cm.get_cmap('tab10')(i / len(unique_labels))
                     for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]

        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_images[:, 0], reduced_images[:, 1],
                    c=point_colors, alpha=0.3, s=10, label='Data Points')

        for i, centroid in enumerate(reduced_centroids):
            plt.scatter(centroid[0], centroid[1], c=[color_map.get(self.cluster_labels.get(i), 'black')],
                        marker='o', s=75, linewidths=1, edgecolors='black', facecolors='white', label=f'Centroid {i}')

        legend_labels = {color_map[label]: f'Label {label}' for label in unique_labels}
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
                          for color, label in legend_labels.items()]
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(handles=legend_patches, title="True Labels")
        plt.show()

    def _plot_3d(self, reduced_images, reduced_centroids, labels, title, xlabel, ylabel, zlabel):
        unique_labels = np.unique(labels)
        color_map = {label: plt.cm.get_cmap('tab10')(i / len(unique_labels))
                     for i, label in enumerate(unique_labels)}
        point_colors = [color_map[label] for label in labels]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_images[:, 0], reduced_images[:, 1], reduced_images[:, 2],
                   c=point_colors, alpha=0.3, s=10, label='Data Points')

        for i, centroid in enumerate(reduced_centroids):
            ax.scatter(centroid[0], centroid[1], centroid[2], c=[color_map.get(self.cluster_labels.get(i), 'black')],
                       marker='o', s=75, linewidths=1, edgecolors='black', facecolors='white', label=f'Centroid {i}')

        legend_labels = {color_map[label]: f'Label {label}' for label in unique_labels}
        legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label)
                          for color, label in legend_labels.items()]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        plt.legend(handles=legend_patches, title="True Labels")
        plt.show()

    def visualize_clusters_tsne_2d(self, mnist_data):
        labels, reduced_images, reduced_centroids = self._prepare_data(mnist_data, TSNE, 2)
        self._plot_2d(reduced_images, reduced_centroids, labels,
                      f'K-Means Clustering of MNIST Data in 2D with t-SNE using {self.n_cluster} clusters', 't-SNE Component 1', 't-SNE Component 2')

    def visualize_clusters_tsne_3d(self, mnist_data):
        labels, reduced_images, reduced_centroids = self._prepare_data(mnist_data, TSNE, 3)
        self._plot_3d(reduced_images, reduced_centroids, labels,
                      f'K-Means Clustering of MNIST Data in 3D with t-SNE using {self.n_cluster} clusters', 't-SNE Component 1', 't-SNE Component 2', 't-SNE Component 3')

    def visualize_clusters_pca_2d(self, mnist_data):
        labels, reduced_images, reduced_centroids = self._prepare_data(mnist_data, PCA, 2)
        self._plot_2d(reduced_images, reduced_centroids, labels,
                      f'K-Means Clustering of MNIST Data in 2D with PCA using {self.n_cluster} clusters', 'Principal Component 1', 'Principal Component 2')

    def visualize_clusters_pca_3d(self, mnist_data):
        labels, reduced_images, reduced_centroids = self._prepare_data(mnist_data, PCA, 3)
        self._plot_3d(reduced_images, reduced_centroids, labels,
                      f'K-Means Clustering of MNIST Data in 3D with PCA using {self.n_cluster} clusters', 'Principal Component 1', 'Principal Component 2', 'Principal Component 3')
