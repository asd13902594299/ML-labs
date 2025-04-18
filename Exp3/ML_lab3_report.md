# Lab3

罗绍玮 2022010749

## 集成学习实现

### Bagging

实现了一个 `BaggingClassifier` 类, 需要指定的构造参数为 `base_estimator` 和 `n_estimators`, 分别代表了使用的基分类器和分类器的个数. 由于 Bagging 的可并行性, 也可以指定 `n_jobs` 参数去控制并行运行的数量, 默认是 `-1` 拉满. 然后预测(`predict`) 时使用的方式是多数投票.

大致实现如下:

```python
class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators, random_state=42, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []

    def _fit_one(self, X, y, indices):
        model = clone(self.base_estimator)
        # Set the model to have different seed
        model.fit(...)
        return model

   	def fit(self, X, y):
        n_samples = X.shape[0]
        indices_list = [np.random.choice(n_samples, n_samples, replace=True)
                        for _ in range(self.n_estimators)]

        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(X, y, indices)
            for indices in indices_list
        )
        
    def predict(self, X_pred):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X_pred)) # shape (n_samples,)
        # Convert to shape (n_estimators, n_samples)
        # Majority vote across classifiers for each sample
        return final_preds
```

### AdaBoosting

实现了一个 `AdaBoostClassifier` 类, 需要指定的构造参数一样为 `base_estimator` 和 `n_estimators`. 由于是多分类任务, 因此训练过程中使用了 `SAMME` 更新的方式. 而预测(`predict`) 时使用每个 `model` 的权重去投票决定.

大致实现如下:

```python
class AdaBoostClassifier:
    def __init__(self, base_estimator, n_estimators=50, random_state=42):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples # init weights

        for _ in tqdm(range(self.n_estimators)):
            # Clone and fit base model
            model = clone(self.base_estimator)
            model.fit(X, y, sample_weight=sample_weights)
            y_pred = model.predict(X)
            incorrect = (y_pred != y).astype(int)
            # Weighted error
            err = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
            if err >= 1 - 1e-10 or err == 0: break
            # SAMME alpha
            alpha = np.log((1 - err) / (err + 1e-10)) + np.log(n_classes - 1)
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Collect weighted votes for each classifier
        class_votes = np.zeros((X.shape[0], len(self.classes_)))
        for alpha, model in zip(self.alphas, self.models):
            y_pred = model.predict(X)
            for i, cls in enumerate(self.classes_):
                class_votes[:, i] += alpha * (y_pred == cls)

        # Choose the class with the maximum vote for each sample
        return np.argmax(class_votes, axis=1)
```

## 实验结果

由于禁止了使用预训练的词向量且我懒得再去用训练集训一个 `word2vec`, 因此选择使用了 `TF-IDF` 的方式进行文本的特征映射.

由于 `KNN` 和 `SVM` 的训练时间太太太长了, 使用 `AdaBoost` 时要跑很长的时间, 因此我选用了 `DecisionTree` 和 `MultinomialNB` 作为两个基分类器. 

### MultinomialNB

在使用 NB 时, 设置 `Bagging: num_base_learners = 16`, `AdaBoost: num_base_learners = 75` 以及不使用集成学习的 accuracy, MAE, 以及 RMSE 的结果如下:

```
Base learner: nb
Ensemble method: none
Accuracy: 0.6014
MAE: 0.7003
RMSE: 1.2793

Base learner: nb
Ensemble method: bagging
Number of estimators: 16
Accuracy: 0.6013
MAE: 0.7011
RMSE: 1.2805

Base learner: nb
Ensemble method: adaboost
Number of estimators: 75
Accuracy: 0.5965
MAE: 0.7161
RMSE: 1.3021
```

可见在使用 75 次迭代的 `AdaBoost` 的情况下, NB 的性能设置比普通单一 NB 还差. 而就算是 `Bagging`, 性能的提升也几乎完全没有. 

这是因为 `MultinomialNB` 是一个低方差, 高偏差的模型, 它是基于特征之间条件独立的假设, 其预测结果通常比较稳定, 不容易受数据扰动的影响. Bagging 中把数据扰动后的模型基本还是一样的, 即投票出来的结果与单一模型差不多. 因此, Bagging 对 NB 几乎不会带来什么改进.

但有趣的是, 若把 `AdaBoost` 的 estimators 数 (num_base_learners ) 拉高, 会得到下面的结果, 迭代次数越高, 指标越好:

```
Base learner: nb
Ensemble method: adaboost
Number of estimators: 300
Accuracy: 0.6179
MAE: 0.6306
RMSE: 1.1907

Base learner: nb
Ensemble method: adaboost
Number of estimators: 1000
Accuracy: 0.6523
MAE: 0.4965
RMSE: 0.9638

Base learner: nb
Ensemble method: adaboost
Number of estimators: 2000
Accuracy: 0.6540
MAE: 0.4898
RMSE: 0.9564
```

这是因为初始时 MultinomialNB 模型能力弱, 无法很好地拟合复杂分布. 而前几轮 Boosting 的效果有限, 甚至会引入更多噪声, 所以 75 个弱学习器时效果反而比原始 NB 差.

不过随着迭代轮数增加, 整体模型就相当于在不断“修补前几轮模型的缺陷”, 最终能逐渐拟合更复杂的边界. 所以对于一下 如 NB 弱模型是需要足够多轮才能有效地 "Boost".

### DecisionTree

一样是设置 `Bagging: num_base_learners = 16`, `AdaBoost: num_base_learners = 75`. 然后 DecisionTree 的 `max_depth` 先设置成了 `3`, 这样能训练的快一些. 得到的结果如下:

```
Base learner: dt
Ensemble method: none
Decision tree depth: 3
Accuracy: 0.5947
MAE: 0.7240
RMSE: 1.3108

Base learner: dt
Ensemble method: bagging
Number of estimators: 16
Decision tree depth: 3
Accuracy: 0.5948
MAE: 0.7239
RMSE: 1.3108

Base learner: dt
Ensemble method: adaboost
Number of estimators: 75
Decision tree depth: 3
Accuracy: 0.6133
MAE: 0.6086
RMSE: 1.1340
```

可见此时的 `AdaBoost` 有效地提升了模型的效果. `Bagging` 一样没啥区别.

但如果我们把 `max_depth` 调成 `10`, 就会得到下面的结果: 
(注意, 这里的 `AdaBoost` 如果用完全的数据集在我笔电上跑了 25分钟...可以透过设置`--data_ratio 0.x` 进行采样) 

```
Base learner: dt
Ensemble method: none
Decision tree depth: 10
Accuracy: 0.6068
MAE: 0.6671
RMSE: 1.2335

Base learner: dt
Ensemble method: bagging
Number of estimators: 16
Decision tree depth: 10
Accuracy: 0.6143
MAE: 0.6576
RMSE: 1.2271

Base learner: dt
Ensemble method: adaboost
Number of estimators: 75
Decision tree depth: 10
Accuracy: 0.6125
MAE: 0.5712
RMSE: 1.0565
```

这时候的 `Bagging` 提高了模型的 Accuracy, 因为模型变复杂, 方差变大所以 Bagging有效了. 但 MAE 并没有下降多少, 反而 `AdaBoost` 在 MAE 指标下依旧有很好的提升.

而同样地, 对 `AdaBoost` 的迭代次数拉高, 模型的各项指标都会上升. 但由于 DecisionTree 的训练时间普遍要比 NB 长, 所以我也不敢把他的模型深度和迭代次数弄得太高, 不然要等他出结果太久了.

```
Base learner: dt
Ensemble method: adaboost
Number of estimators: 75
Decision tree depth: 3
Accuracy: 0.6133
MAE: 0.6086
RMSE: 1.1340

Base learner: dt
Ensemble method: adaboost
Number of estimators: 150
Decision tree depth: 3
Accuracy: 0.6227
MAE: 0.5858
RMSE: 1.1053

Base learner: dt
Ensemble method: adaboost
Number of estimators: 75
Decision tree depth: 2
Accuracy: 0.6095
MAE: 0.6512
RMSE: 1.2095

Base learner: dt
Ensemble method: adaboost
Number of estimators: 150
Decision tree depth: 2
Accuracy: 0.6193
MAE: 0.6253
RMSE: 1.1764

Base learner: dt
Ensemble method: adaboost
Number of estimators: 300
Decision tree depth: 2
Accuracy: 0.6273
MAE: 0.6055
RMSE: 1.1522
```

## 总结

Bagging 主要是降低方差, 但 NB 本身方差就小, 所以没啥用; 而在DT中, 只有模型变复杂了方差大了才会开始见效.

Boosting 能逐步纠正偏差高模型的缺点, 但需要足够多轮才能发挥效果; 是一个不断学习残差的过程, 迭代越多, 修正得越彻底, 尤其适合弱模型, 因此适合修补小决策树的偏差; 但需要注意的是轮数过多可能会过拟合.

最后是一个对应个四个组合的小总结:

| 模型类型                | Bagging 效果 | AdaBoost 初期 | AdaBoost 后期       | 原因                                                         |
| ----------------------- | ------------ | ------------- | ------------------- | ------------------------------------------------------------ |
| MultinomialNB           | 几乎无提升   | 有时更差      | 效果明显提升        | Bagging 无法降低低方差模型误差; Boosting 能逐步弥补偏差      |
| DecisionTree (depth=3)  | 无明显提升   | 马上改善      | 继续变好但幅度较小  | DT 的弱模型属性和 Boosting 完美契合                          |
| DecisionTree (depth=10) | 有小幅提升   | 明显改善      | 待观察 (受限于时间) | 模型复杂度提升, Bagging 有效; Boosting 依然有用但计算代价变大 |

同时, 由于使用了 `TF-IDF` 而非 `word2vec`, 文本中并未考虑到词"顺序"和词"组合"的特征. 因此若使用了 `word2vec` 可能可以更有效. 此外, 在 `predict` 中除了使用"多数投票"外, 还可以尝试使用其他如平均上取整等的预测方式. 但由于时间关系和分析的难度, 故未做额外的评估.