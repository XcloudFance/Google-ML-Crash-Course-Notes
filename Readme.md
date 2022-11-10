# Google Crash Course - Machine Learning

# Tensorflow

tf.keras - 最高层 - 里面包含一些已经存在的模型

### batch & epoch & iteration

深度学习优化算法 - 梯度下降

每次参数更新有两种形式

1. 遍历全部数据集，并且一次性算出损失函数，然后算函数对各个参数的梯度，更新梯度
    1. 这种方法计算速度慢，不支持在线学习
    2. 每次更新参数都要遍历一次数据集
    3. 这种方法叫batch gradient descent 批梯度下降
2. 每看一次数据就更新一次损失函数，求梯度更新参数，称为随机梯度下降(stochastic gradient descent)
    1. 速度快，但是难收敛，可能几次都到不了最低点，造成函数震荡比较剧烈

为了克服两种方法的缺点，现在一般采用的是一种折中手段，**mini-batch gradient decent**，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大

现在用的优化器SGD是stochastic gradient descent的缩写，但不代表是一个样本就更新一回，还是基于mini-batch的。

那 batch epoch iteration代表什么呢？

> （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练；
> 
> 
> （2）iteration：1个iteration等于使用batchsize个样本训练一次；
> 
> （3）epoch：1个epoch等于使用训练集中的全部样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次。
> 

比如训练集有500个样本，batchsize = 10 ，那么训练完整个样本集：iteration=50，epoch=1.

batch: 深度学习每一次参数的更新所需要损失函数并不是由一个数据获得的，而是由一组数据加权得到的，这一组数据的数量就是batchsize。

batchsize最大是样本总数N，此时就是Full batch learning；最小是1，即每次只训练一个样本，这就是在线学习（Online Learning）。当我们分批学习时，每次使用过全部训练数据完成一次Forword运算以及一次BP运算，成为完成了一次epoch。

### representation 表达法

机器学习专注于表示法，通过添加和改善特征来调整模型

定义： 将**原始数据**转换为**特征矢量**

> 这边有点类似将字符串以0，1，2，3的数字形式进行表示，但是数字和数字之间的特征转换其实没有意义，直接把数字当成权值带入就行
> 

### 映射分类值

比如说不同街道 -> 指定编码

也有可能使用一个一维向量来表示那一类

[0,0,1,1,0,0]

**one-hot encoding**: 只可能属于其中一个，只有一个1，比如房屋属于某个城市

**multi-hot encoding:** 可能属于多个1，比如房屋处于两个街道的交叉点

有点像oi比赛里面常出现的对不同的特殊数据进行特征化处理

- 当分类很多时，可以采用稀疏表示(sparse representation)
- 避免很少使用离散特征值
    - 特征值至少出现5次，有助于机器学习了解特征
- 要有清晰的定义
    - 引入项目编号是无意义的
    

## Feature Crosses

<aside>
💡 特征交互也叫特征组合，通过将两个或多个特征相乘，来实现对样本空间的非线性变换，增加模型的非线性能力。

</aside>

- 特征交叉使用 [A x B]来进行表示
- 也可以是[A x B x C x D]
- 如果是布尔/二进制的情况，也可以单纯用A B 表示
- 这样的学习可以很好的扩展到海量数据
- 但是如果没有特征交叉，这些模型的表现力将受到限制
- 使用特征交叉➕海量数据是学习高度复杂模型的一个很有效的策略
- 深度学习-神经网络也是另一种海量学习的好方法（伏笔）

### kinds of features crosses

E.g.

- [A x B] : 这是一个乘了两个值的feature cross
- [A x A]: 这是一个通过对单个特征进行平方而形成的feature  cros

## 正则化  Regularization for simplicity

防止过耦合，降低模型复杂度而引入的计算因子。

可以通过正则化率来调整模型中所有特征的权重的重要性

降低复杂模型的复杂度防止过拟合（overfitting）

### 方法1:  early stopping

在训练数据的结果实际收敛(converge)之前停止训练

尽量到达红色曲线(Validation data)的底端

> 因为红色曲线会到底端之后反过来上升loss，原因就是过拟合导致的训练损失（training loss）
> 

限制步数和学习速率

### 方法2:  Penalizing Model Complexity

- 避免模型的复杂性

$$
minimize: Loss(Data | Model) + complexity(Model)
$$

- 既要确保正确的训练数据，又不能过度信赖训练数据以免模型过于复杂

## 定义模型复杂度(defining model complexity)

- 采用较小的权重，小到几乎可以忽略，并且能正确获取训练样本
- **L1 Regularization:**
    - 稀疏性正则化 Regularization for Sparsity
    - L1 会降低 Weight的绝对值
    - L1的导数为k（为一个常数，和权重无关）
    - 将无意义的权重减少到0

- **L2 Regularization(known as ridge):**
    - complexity = sum of squares of the weights
    - L2 会降低权重的平方
    - 结果上来说，L2和L1的导数不相同
        - L2的导数是2*weight
    - 会惩罚过大的权重(tend to force bigger weight to drop more quickly than the smaller weight)
    - 对于线性的模型来说会更偏爱更平的斜率
    - 贝叶斯先验概率 - Bayseian Prior:
        - weights should be centered around zero
        - weights should be normally distributed
        - Loss function with L2 regularization
            
            $$
            Minimize(Loss(Data|Model) + \lambda(w_{1}^{2} + ... + w_{n}^{2}))
            $$
            

<aside>
💡 这里的lambda表示用来控制权重平衡方式的值

</aside>

- 如果有大量的数据，且训练数据和测试数据差不多，可能不需要多少正则化
- 如果把系数直接设置到了-6，可能也不需要正则化
- 但是如果训练数据不多，和测试数据又不大一样，这时候就需要大量的正则化
- 可能需要交叉验证或者单独测试集进行调整

> 现在不仅仅是最小化损失，还是在最小化损失+复杂性，称为**结构风险最小化**
> 

### Lambda 的调整

执行L2正则化对模型的影响

- 鼓励权重的值朝0，但不完全为0
- 鼓励权重的均值接近0，正态分布

<aside>
💡 如果lambda太高，模型就会很简单，有可能拟合不足

</aside>

<aside>
💡 如果lambda太低，会导致模型变得复杂，可能会出现过拟合

</aside>

## Logistic Regression 逻辑回归

核心：输出概率

‘s’ 型函数：

x: 提供线性模型

$$
Z = w^{T}x + b
$$

 

$$
y = \frac{1}{1 + e^{-z}}
$$

E.g.

p(house will sell) * price = expected outcome

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled.png)

如果z表示用逻辑回归训练的模型的线性层的输出

则S型的函数会生成一个0-1之间的一个值（概率）

### 对数损失 LogLoss

$$
LogLoss = \sum_{(x,y)\in D} -ylog(y') - (1-y)log(1-y')
$$

- (x,y) 属于D是包含很多有标签样本(x,y)的数据集
- y是由标签样本的标签，由于是逻辑回归，每个y必须是0/1
- y’是对于特征集x的预测值(介于0到1之间)
- 正则化对于逻辑回归非常重要，在高纬度下会不断尝试损失到0
    - 使用L2正则
    - 早停法
- 高效，训练和预测速度非常快

<aside>
💡 LinearRegressor 用的是L2损失，所以在输出解读为概率的时候，并不能有效的惩罚误分类，0.9和0.9999的负分类样本是否被分类为正分类，差异很大，但是L2损失不会区分这些状况，相比之下对数损失对这些错误惩罚力度更大

</aside>

## 分类 - Classification

- 分类与回归
    - 有时候会针对逻辑输出使用逻辑回归，介于(0,1)
    - 其他时候会对离散二元分类的值设定阈值
    - 阈值可以进行调整
    
- 准确性可能会造成误导
    - 特别是在不同类型的错误有不同的代价的时候
    - 特别是正分类和负分类极度不平衡的时候

- 预测和实际有四种情况(真正例，假正例，假负例，真负例）
    - TP（预测正确）True positive
    - FP
    - TN（预测正确）
    - FN

- 精确率(precision)，准确率(accuracy)和召回率(recall)
- 

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Accuracy = \frac{TP + TN}{TP+TN+FP+FN}
$$

$$
Recall = \frac{TP}{TP+FN}
$$

当使用分类不平衡的数据集，准确率会误导

精确率是所有识别为正的样本中，实际为正的样本比例

召回率是，所有正样本中，被正确识别出来的比例(它计算的是所有检索到的item占所有"应该被检索到的item"的比例。)

### ROC 曲线

一种分析模型在所有分类阈值下的效果的图标

每个点的坐标都是一个决策阈值的TP率（Y 坐标）和FP率（X 坐标）

AUC： ROC曲线下的面积

也就是说曲线下面积测量是从(0,0)到(1,1)之间整个roc曲线以下的整个二维面积

<aside>
💡 实用特点：曲线下面积的尺度不变。它测量的预测排名的情况，而不是测量绝对值。 曲线下的面积分类阈值不变，它测量模型预测的质量，而不考虑所选的分类阈值

</aside>

### Prediction Bias 预测偏差

如果出现观测的平均值和预测的平均值不一样，说明出现了预测偏差

PB = 预测平均值 - 数据集中相应标签的平均值

- 作用：
    - 如果出现非常高的PB，说明模型有地方错误，因为这表明模型对正类比标签出现概率预测有误
- 产生原因：
    - 特征集不完整
    - 数据集混乱
    - 模型实现流水线中有错误
    - 训练样本有偏差
    - 正则化过强
        
        

不要在校准层修复偏差，可以在模型修复

<aside>
💡 对于抛硬币这种不是0就是1的，如果预测出来概率是0.5 0.3之类的，肯定要进行校准，不然数据没意义，因此只有足够多的平均值预测时，平均预测和平均观测值才会有意义

</aside>

### 调优

- 不过拟合，就训练更长时间
- 可以增加步数/批量大小
- 所有指标同时提升，这样损失指标就能够很好的代理AUC和准确率了

## Regularization: Sparsity

将噪声系数直接置为0， known as L0 Regularizaiton

只会惩罚没有置为0的系数

### 考虑一下Feature Crosses(特征交叉)

- 稀疏的特征交叉可能会显著增加特征空间
- 导致 模型大小(RAM) 可能会变得很大
- 也有可能导致过拟合（因为噪声系数）

### Relax to L1 regularization

- 比起L2， L1会更提倡稀疏度
- 惩罚权重的和
- L0比L1更极端，参数更容易被惩罚为0
- 如果要从10k个要素里面剩余10个要素，可能需要尝试L0正则化

## 神经网络

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled%201.png)

### 神经网络的标准组件

- 一组节点，类似于神经元，位于居中
- 一组权重，表示每个神经网络层与下方层的联系
- 一组偏差，每个节点一个偏差
- 一个激活函数，激活函数可以对每个节点的输出进行转换，不同的层可以有不同的激活函数

### Relu Function - 一种人工神经网络中常用的激励函数

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled%202.png)

> **整流线性单位函数**
（Rectified Linear Unit, **ReLU**），又称**修正线性单元**，是一种[人工神经网络](https://zh.m.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)中常用的激励函数（activation function），通常指代以[斜坡函数](https://zh.m.wikipedia.org/wiki/%E6%96%9C%E5%9D%A1%E5%87%BD%E6%95%B0)及其变种为代表的非线性函数。
> 

### back propagation 反向传播

- 是一种另类的梯度下降法
- 可以用链式法对网络的每层权重为变数计算损失函数的梯度，来更新权重，最小化损失函数

例子：[Back Propagation（梯度反向传播）实例讲解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40378224)

- prerequisite: 函数必须是可导的，有一两个不连续的地方是允许的
- 梯度可能会消失，每个额外的训练层都可能减少信号/噪声
- 这里使用Relu函数会变得非常有用
- 如果训练太快，梯度可能会爆炸，导致模型出现NaN的情况，这时候就需要降低学习速率
- Relu函数可能会失效
    - 因为我们硬性地将上限设置为0，最终的内容都将归为0，梯度无法进行反向传播，就永远无法返回Relu层所在的位置

### graph of two-layer model

![2hidden.svg](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/2hidden.svg)

尽管有很多层的覆盖，但是由于模型本身就是线性的，所以输入输出也会是通过线性计算

![activation.svg](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/activation.svg)

直到现在，通过添加了一层非线性的transformation layer，整个模型变得非线性了

## Multi-Class Neural Sets

- 逻辑回归对于两极化的结果非常有用
    - 是否为垃圾邮件
    - 点击或者没有点击
- 那如果是多元的情况，那就需要用到多层的神经集合

### One-Vs-All Multi-class

- 为每个class创建一个独特的输出
- 可以在单独的一个深度学习网络内执行，或者是分别的不同的模型
- 只需要让模型的开头具有不同的输出节点，并通过模型的其余部分共享内部表示法，用高校的方式同时对这些点进行训练

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled%203.png)

### softmax 函数（归一化）

- 逻辑回归 generalize to one more class
- 适用于multi-class的情况

> softmax 函数将 K 个实数的向量*z*作为输入，并将其归一化为由输入数字的指数成比例的*K*个概率组成的[概率分布](https://en.wikipedia.org/wiki/Probability_distribution)。也就是说，在应用 softmax 之前，某些矢量分量可能是负数或大于 1;并且可能总和不等于 1;但是在应用SoftMax之后，每个组件都将在[间隔](https://en.wikipedia.org/wiki/Interval_(mathematics))中 分量加起来为 1，以便可以解释为概率。此外，较大的输入分量将对应于较大的概率。
> 

## Embeddings 嵌入

<aside>
💡 这一部分和推荐系统非常相关，可以说推荐系统很大一部分的训练方法和embeddings有关

</aside>

输入：100w个电影和50w个用户已经选择了要看的电影

输出: 给这些用户推荐电影

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled%204.png)

Embedding 和 相似度有着很大的关系，通过相似度进行推荐内容并不少见

这里开始讲述对1维数据进行分类不同的电影相似进行推荐

如果用简单的动漫和真人分类，可能会出现<<Incredibles>>其实不适合小孩子看的情况

所以一维的分类有失偏颇，我们需要更多维的学习

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled%205.png)

现在是2d了，相比于之前在同一行排列的数据，现在这些数据都有了自己的xy点，如果把blockbuster类型的往上面放，然后art house类的往下面放

这样就能选出适合推荐了

可以通过上中下的形式去推荐，体现了2d的好处

![Untitled](Google%20Crash%20Course%20-%20Machine%20Learning%200abc78e7a1c5495b96e9c4685384dba6/Untitled%206.png)

更复杂了.

### D-Dimensional Embeddings

- 假设用户的电影兴趣可以被切分成d个方面
- 每个电影都可以有自己的d维的点，可以表现出这个电影是否符合某一方面
- 可以通过数据学习

### Learning Embeddings in a deep network

- 可以不用区分开训练过程，可以让Embedding layer直接变成其中一个hidden layer作为一个维度
- （说明有多个维度就有多个hidden layer)
- 然后用反向传播，做权重调整
- *监督学习
- 协同过滤：让用户一起参与过滤过程
- 

# References

- [(105条消息) AI_铭记北宸的博客-CSDN博客](https://blog.csdn.net/hanjingjava/category_8118993.html)
- [https://developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)/
- [Softmax function - Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)