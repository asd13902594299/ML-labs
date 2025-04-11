# Lab1

罗绍玮 2022010749

## 实现 Bayes Classifier

(使用 `python3 main.py` 运行)

实现了一个最朴素的 classifier.

其中的参数只有以下六项: 

- `total_map`, `total_count`: train 给的所有数据的词频表以及所有字的出现次数;
- `ham_map`, `ham_count`:  train 给的所有"tag为ham"的数据的词频表以及所有字的出现次数;
- `spam_map`, `spam_count`: train 给的所有"tag为sham"的数据的词频表以及所有字的出现次数;

#### 预处理

无视 Header 中的信息, 直接提取正文的内容继续训练.

之后再以各标点符号(除了 "!" 和 "''"), 空格, 换行等作为分隔符对该字串进行分词. 同时, 需要将乱码的东西转换成 "�".

最后再把每个词换成小写以及去除所有空白字符.

最终一个 data 对于 classifier 而言会长这样: `["irngwzbeb","fordpersonnel","com","for","your","family", "�"...]`. 

分好词之后还需要到 `index` 文件中把该 data 对应的 `tag` 找出来. 这样后续才能更方便地进行 train 和 validate.

#### Training

数据预处理之后, 便是进入到了 training 的阶段. 

Train 的方式就是把上面处理好的 `list[str]` 进行一个遍历, 遍历的过程中一定会把当前 `str` 加入到 `totoal_map` 以及 `total_count`, 此外会根据 `tag` 选择加入到 `spam/ham_count/map` .

#### Predicting

其 predict 的方式使用了课件中给出的最大后验估计公式:

```python
def predict(self, data: list[str]) -> int:
    ham_prob = self.ham_count / self.total_count
    spam_prob = self.spam_count / self.total_count
    for word in data:
        ham_prob *= self.ham_map.get(word, 0) / self.ham_count
        spam_prob *= self.spam_map.get(word, 0) / self.spam_count
    return 0 if ham_prob > spam_prob else 1
```

#### Result

对所有的 data 使用一次 5-折交叉验证, accuracy 等信息如下:

```
Fold 0 - Accuracy: 0.7155
Fold 0 - Precision: 0.7174, Recall: 0.9963, F1-score: 0.8342
Fold 1 - Accuracy: 0.7209
Fold 1 - Precision: 0.7227, Recall: 0.9965, F1-score: 0.8378
Fold 2 - Accuracy: 0.7217
Fold 2 - Precision: 0.7238, Recall: 0.9960, F1-score: 0.8384
Fold 3 - Accuracy: 0.7146
Fold 3 - Precision: 0.7166, Recall: 0.9961, F1-score: 0.8335
Fold 4 - Accuracy: 0.7173
Fold 4 - Precision: 0.7188, Recall: 0.9971, F1-score: 0.8354
Average Accuracy: 0.7180
Overall Precision: 0.7199, Recall: 0.9964, F1-score: 0.8358
```

可以发现模型的 Accuracy 仅有 **72% **不到, 但 recall 相当高. 这不难理解, 是因为下面 Q2 中的 0 概率问题导致的. 一旦出现了训练集的没有的数据时, `ham_prob` 或 `spam_prob` 的概率就会直接变成 0, 从而导致预测的结果一定是 `1(spam)`, 即 classifier 会很倾向给出 spam 的预测结果.

## Q1

(使用 `python3 main.py Q1.{1,2,3}` 运行)

对 Q2 中使用了 smooth 的 classifier: 

使用 5% 的数据进行训练并用余下 95% 的数据进行验证的结果:

```
Accuracy: 0.9498, Precision: 0.9866, Recall: 0.9621, F1-score: 0.9742
```

使用 50% 的数据进行训练并用余下 50% 的数据进行验证的结果:

```
Accuracy: 0.9281, Precision: 0.9946, Recall: 0.9327, F1-score: 0.9627
```

使用 100% 的数据进行训练并用 100% 的数据进行验证的结果:

```
Accuracy: 0.9470, Precision: 0.9958, Recall: 0.9508, F1-score: 0.9728
```

可以发现在仅使用 5% 训练样本的时候效果最好.

一般而言, 增大训练集会带来更好的泛化能力和更稳定的预测效果. 能从 50% 测试数据集增长到 100% 时, 其所有指标都明显上升看出.

然而在 5% 的情况并非如此. 猜测原因如下:

- 5% 训练数据恰好是"优质的数据”(正负样本比例均衡, 垃圾邮件和正常邮件的特征分布典型), 此时即使数据量少, 模型仍然能很好地概括测试数据; 而 50% 训练数据可能引入了一些“噪声”, 导致模型学习到了不稳定的模式, 从而使得 Recall 下降.
- 增加了数据, 但质量不均衡(如某类别的数据增长较多, 导致词频统计更倾向于该类别), 影响了模型的泛化能力.

## Q2

(使用 `python3 main.py Q2` 运行)

如果在训练集中没有样本出现, 则 $P(x_i=k∣y=c)=0$, 则 $P(y=c∣x_1,…,x_i=k,...,x_n) = 0$. 

这问题最直接的体现就是, 无论其他的特征有多么的支持 $y=c$, $P(y=c∣x_1,…,x_n)$ 都必然会是 $0$, 从而导致错误的 prediction. 此外, 这个问题如果是体现在使用 log 计算的话, 会导致 error, 毕竟 log 的定义域中不含 $0$.

而在下面实现当中, 使用的就是 PPT 中提到的(Laplace)平滑, 即在分子上$+\alpha$, 分母上$+\alpha \times |V|$, 其中的 $V$ 是训练数据中出现过的单词数量. 该平滑保证了分子不会出现 $0$, 且分母上的变化也保证了所有的 probabilities 相加起来依旧为 $1$. 推导如下:
$$
\begin{align}
\sum_{x_i\in V} P(x_i|y=c) &= 1 \\

\sum_{x_i\in V} \hat{P}(x_i|y=c) &= \sum_{x_i\in V} \frac{count(x_i,c)+\alpha}{\sum_{x_j\in V}count(x_j,c)+\alpha|V|} \\
&= \frac{\sum_{x_i\in V}count(x_i,c)+\sum_{x_i\in V}\alpha}{\sum_{x_j\in V}count(x_j,c)+|V|} \\
&= \frac{\sum_{x_i\in V}count(x_i,c)+\alpha|V|}{\sum_{x_j\in V}count(x_j,c)+\alpha|V| }\\
&= 1 \\
&= \sum_{x_i\in V} P(x_i|y=c)
\end{align}
$$

### 实现

首先是对模型添加了变量, `vocab_appear`, `vocab_count` 记录下 train 中出现过的单词以及其数量用于 smoothing.

然后 train 的过程中需要更新 `vocab_appear`, `vocab_count`.

最后把 predict 改成:

```python
 def predict(self, data: list[str]) -> int:
    # using log addition to do the prediction
    ham_prob = log(self.ham_count / self.total_count)
    spam_prob = log(self.spam_count / self.total_count)
    for word in data:
        # using Laplace smoothing
        ham_prob += log((self.ham_map.get(word, 0) + 1) / (self.ham_count + self.vocab_count))
        spam_prob += log((self.spam_map.get(word, 0) + 1) / (self.spam_count + self.vocab_count))
    return 0 if ham_prob > spam_prob else 1
```

### Result

 5-折交叉验证 accuracy 等信息如下:

```
Fold 0 - Accuracy: 0.9433
Fold 0 - Precision: 0.9943, Recall: 0.9484, F1-score: 0.9708
Fold 1 - Accuracy: 0.9401
Fold 1 - Precision: 0.9947, Recall: 0.9449, F1-score: 0.9691
Fold 2 - Accuracy: 0.9402
Fold 2 - Precision: 0.9954, Recall: 0.9444, F1-score: 0.9692
Fold 3 - Accuracy: 0.9405
Fold 3 - Precision: 0.9932, Recall: 0.9466, F1-score: 0.9693
Fold 4 - Accuracy: 0.9344
Fold 4 - Precision: 0.9955, Recall: 0.9384, F1-score: 0.9661
Average Accuracy: 0.9397
Overall Precision: 0.9946, Recall: 0.9445, F1-score: 0.9689
```

可以发现模型的性能还不错, 但不同指标的高低反映了模型在不同方面的表现. 比如相当高的 Precision 但 Recall 略低. 这意味着模型倾向于保守地预测垃圾邮件, 更愿意把邮件分类为正常邮件而是不随意标记垃圾邮件. 同时在 5 轮交叉验证中, Accuracy 在 **93% - 94%** 之间浮动, 但整体还是稳定, 没有过大的方差.

## Q3

(使用 `python3 main.py Q3` 运行)

除了词袋模型(frequence analysis)之外, 还可以使用以下特征去提高侦测效能:

- Header 信息: 

  在 Mail Sender 中, 一些特定的组合或域名的邮箱会是常见的垃圾邮件来源, 因此可以用作一个训练的特征. 此外, Subject 中的信息也应当影响邮件的重要程度. 我的实现中也有类似的做法.

  除了以上两个 field 外, Header 中的如 Time, Priority 等信息也可以帮助分析是否垃圾邮件, 但我个人感觉他们的重要性没有 Sender 和 Subject 重要, 因此我并没考虑使用他们

### 实现

基于 Q2 中使用了平滑的 classifier.

方法是提取 Header 中的 "Subject" 和 "Sender"  的内容, 然后把他们与正文进行 concat.

同时, 我还把文本中的纯数字转换成 `/NUM1`, `/NUM2`, `/NUM3`, `/NUM3+`, 以为着该数字的长度, 这样我感觉能更有效地使用"数字"的特征.

### Result

 5-折交叉验证 accuracy 等信息如下:

```
Fold 0 - Accuracy: 0.9467
Fold 0 - Precision: 0.9957, Recall: 0.9506, F1-score: 0.9726
Fold 1 - Accuracy: 0.9447
Fold 1 - Precision: 0.9951, Recall: 0.9491, F1-score: 0.9716
Fold 2 - Accuracy: 0.9445
Fold 2 - Precision: 0.9957, Recall: 0.9484, F1-score: 0.9714
Fold 3 - Accuracy: 0.9428
Fold 3 - Precision: 0.9932, Recall: 0.9489, F1-score: 0.9705
Fold 4 - Accuracy: 0.9331
Fold 4 - Precision: 0.9965, Recall: 0.9362, F1-score: 0.9654
Average Accuracy: 0.9424
Overall Precision: 0.9952, Recall: 0.9466, F1-score: 0.9703
```

比起不使用这些信息的 Q2 有了小许提升.
