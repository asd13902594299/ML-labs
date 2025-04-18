# Ensemble Learning

罗绍玮 2022010749

## 文件树结构

```
.
├── AdaBoost.py
├── Bagging.py
├── README.md
├── data
│   └── exp3-reviews.csv
├── main.py
├── models
│   └── *.pkl
├── requirements.txt
├── run.sh
└── utils.py
```

- `data`: 数据集.
- `models/*`: 模型保存的地方.
- `AdaBoost.py`: AdaBoostClassifier 集成学习算法的实现.
- `Bagging.py`: BaggingClassifier 集成学习算法的实现.
- `utils.py`: 加载数据/统计准确率等的 Helper functions.
- `main.py`: 用于运行的 py 主文件. 
- `run.sh`: 基于 `main.py` 的脚本运行文件.

## 使用方式

使用 `pip install -r requirements.txt` 建立运行所需环境.

在目录下创建一个 `models` 文件夹, 然后在运行时 `main.py` 选择参数:

- `--num_base_learners`: 集成算法的 estimator 数量. (必填)
- `--base_learner`: 使用什么基分类器`{'nb', 'dt', 'none'}`, `none` 意味着直接使用单一基分类器. (必填)
- `--ensemble_method`: 使用什么集成算法`{'bagging', 'adaboost', 'none'}`, `none` 意味着直接使用单一基分类器. (必填)
- `--dt_depth`: `DecisionTree` 分类器的最大深度. 只有在使用 `dt` 作为基分类器时有意义.
- `--seed`: 运行时的随机数.
- `--num_features`: 限制 `TF-IDF` 的 feature 数. 默认 5000
- `--data_ratio`: 采用数据集的百分比. 默认使用全部, 1.0
- `--n_jobs`: 并行数量. 只有在使用 `bagging` 作为集成算法时有意义.
- `--load_model`: Enable 加载模型.
- `--save_model`: Enable 保存模型.