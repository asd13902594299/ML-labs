import struct
import numpy as np


def load_mnist_data(image_file, label_file) -> list[tuple[int, np.ndarray]]:
    """ 
    Load MNIST data from image and label files, 
    and return a list of tuples containing the label and image matrix(28x28).
    """

    with open(image_file, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(label_file, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return [(labels[i], images[i]) for i in range(num)]  # Pair labels with images


def accuracy(classifier, mnist_data):
    correct = 0
    for label, image in mnist_data:
        if classifier.predict(image) == label:
            correct += 1
    return correct / len(mnist_data)


if __name__ == '__main__':
    # Load image data as a list of matrices
    label_path: str = "data/MNIST/raw/train-labels-idx1-ubyte"
    image_path: str = "data/MNIST/raw/train-images-idx3-ubyte"
    image_matrices = load_mnist_data(image_path, label_path)

    np.set_printoptions(linewidth=120, threshold=1000, edgeitems=10)

    # Print shape of first matrix
    print(f'First image label: {image_matrices[0][0]}')
    print(f"First image matrix shape: {image_matrices[0][1].shape}")
    print(image_matrices[0][1])  # Show first image matrix as an example
