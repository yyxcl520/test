# 2.1 数据集加载
from sklearn.datasets import load_iris # 导入鸢尾花数据集
iris = load_iris() # 载入数据集
print('iris数据集特征')
print(iris.data[:10])

print('iris数据集标签')
print(iris.target[:10])

# 2.2 模型加载
from sklearn import tree # 导入决策树包
clf = tree.DecisionTreeClassifier() #加载决策树模型

# 2.3 模型训练
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集

# 2.4 模型预测
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
print('\n预测结果前10个样本:')
print(predictions[:10])

# 2.5 结果评估
from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('\nAccuracy:%s'% accuracy_score(iris.target[120:], predictions))

# 2.6 决策树调参
# 使用entropy作为criterion参数
print('\n使用entropy作为criterion参数:')
clf_entropy = tree.DecisionTreeClassifier(criterion='entropy')
clf_entropy.fit(iris.data[:120], iris.target[:120])
predictions_entropy = clf_entropy.predict(iris.data[120:])
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions_entropy))

# 设置max_depth参数
print('\n设置max_depth=2:')
clf_depth = tree.DecisionTreeClassifier(max_depth=2)
clf_depth.fit(iris.data[:120], iris.target[:120])
predictions_depth = clf_depth.predict(iris.data[120:])
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions_depth)) 