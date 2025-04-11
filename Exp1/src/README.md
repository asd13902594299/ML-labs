# NaiveBayes

罗绍玮 2022010749

## 文件树结构

```
.
├── NaiveBayes.py
├── README
├── cross_validation
│   └── cv*
├── data  
│   └── *
├── label
│   └── index
├── main.py
└── utils.py
```

- `data`, `label`: 数据
- `cross_validation`: 放置预处理后用于 train 和 validate的数据
- `NaiveBayes.py`: 朴素贝叶斯邮件分类器的实现

- `utils.py`: 预处理以及一些 helper 的函数实现
- `main.py`: 用于运行的 py 文件

## 使用方式

在目录下创建一个 `cross_validation` 文件夹, 然后根据输入参数运行对应的任务:

- 不帶平滑的分類器5-折交叉验证的结果: `python3 main.py`
- Q1 中使用帶平滑的分類器 5% 训练数据的结果: `python3 main.py Q1.1`
- Q1 中使用帶平滑的分類器 50% 训练数据的结果:  `python3 main.py Q1.2`
- Q1 中使用帶平滑的分類器 100% 训练数据的结果:  `python3 main.py Q1.3`
- Q2 使用帶平滑的分類器5-折交叉验证的结果:  `python3 main.py Q2`
- Q3 使用帶平滑且有額外特徵的分類器5-折交叉验证的结果:  `python3 main.py Q3`