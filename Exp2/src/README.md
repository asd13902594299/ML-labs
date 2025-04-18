# K-Means

罗绍玮 2022010749

## 文件树结构

```
.
├── ClusterVisualizer.py
├── Kmeans.py
├── README
├── data
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-labels-idx1-ubyte
│           ├── train-images-idx3-ubyte
│           └── train-labels-idx1-ubyte
├── main.py
├── models
│   └── kmeans_cluster{xx}.pkl
└── utils.py
```

- `data/MNIST/raw/*`: 数据集.
- `models/*`: 模型保存的地方.
- `Kmeans.py`: K-means 聚类算法实现.
- `utils.py`: 加载/预处理数据/统计准确率等的 Helper functions.
- `main.py`: 用于运行的 py 文件. 

## 使用方式

使用 `pip install -r requirements.txt` 建立运行所需环境.

在目录下创建一个 `models` 文件夹, 然后在 `main.py` 中的 `args` 控制以下超參:

- `n_cluster`: 聚类的 cluster 数量.
- `load_from_path`: 决定了是否跳过训练阶段从而直接从 path 中加载模型. 默认是 `"models/kmeans_cluster{n_cluster}.pkl"`
- `visualize_dim`: 可视化降维的维数. (2, 3)
- `visualize_alg`: 可视化降维使用的算法. (PCA, TSNE)

