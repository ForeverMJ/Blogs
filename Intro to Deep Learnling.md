# Intro to Deep Learnling
### KEY WORDS

Keras and Tensorflow

deep neural networks.

**regression** and **classification（回归和分类）**

**stochastic gradient descent（梯度下降）**

### **What is Deep Learning?**

近些年，人工智能领域中令人最印象深刻的进步来自于深度学习。自然语言处理，图像识别，还有游戏领域（围棋等）甚至达到或超过了人类水平的表现。所以到底什么是深度学习呢？深度学习是机器学习的一种方法，特点是deep stacks of computations。深度学习使用的基本模型是神经网络，他的灵感来源自人脑的基本结构和功能，每个神经元负责简单的计算，但通过其连接可以形成复杂的模式。

# Chapter1 A Single Neuron（神经元）

### **The Linear Unit**

!https://storage.googleapis.com/kaggle-media/learn/images/mfOlDR6.png

                                                            *The Linear Unit: y=wx+b*

上图是一个神经元模型：

x是输入。

w是权重。神经网络通过不断修改权重来学习。实际上训练的过程就是根据预测结果和实际结果的误差调整权重。这是通过反向传播算法和梯度下降优化算法完成的。

b是一种特殊的weight，我们称作bias（偏差）。它指的是模型预测和真实值之间的差别。具体体现在训练集和测试集上的预测误差。当偏差较高时，通常意味着模型比较简单，无法捕捉到数据的复杂性，而导致欠拟合（Underfitting）。

y是神经元最后的输出。

### **Example - The Linear Unit as a Model**

!https://storage.googleapis.com/kaggle-media/learn/images/yjsfFvY.png

                                                        *Computing with the linear unit.*

上面的图片作为例子。实际上作为一个linear unit去进行计算。如果让sugar等于5作为input的话，那么output就是102.5

### **Multiple Inputs**

!https://storage.googleapis.com/kaggle-media/learn/images/vyXSnlZ.png

                                                        *A linear unit with three inputs.*

当然模型可以变得更复杂一点。我们可以增加更多的input给这个神经元，然后给每个input不一样的权重，然后把他们相加在一起，输出。

### How to Create  Linear Units in Keras

最简单的方式是通过keras.Sequential。

简单介绍一下keras，它是一个高级神经网络API，能够运行在TensorFlow等深度学习框架上。提供了构建和训练深度模型的简便方法。

`keras.Sequential`是Keras中用于创建模型的一个类。它表示神经网络的线性堆叠，允许用户逐层构建模型。每一层依次添加到模型中，形成一个前馈神经网络。

定义有三个输入特征【sugars，fiber，protein】，一个输出【calories】的神经元：👇

```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unitmodel = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

 

https://www.kaggle.com/code/ryanholbrook/a-single-neuron?scriptVersionId=126574232&cellId=3

# Chapter2 **Deep Neural Networks**

刚才介绍了一个神经元的构建。那么我们如果从仿生的角度来看，一定要实现利用多个神经元进行复杂的计算。现在我们就来看一下如何结合和修正一个神经元去模块化更复杂的功能（The key idea here is *modularity*）。

### **Layers**

神经网络通常将它们的神经元组织成层。当我们将一组具有共同输入的线性单元收集在一起时，我们就得到了一个密集层。

!https://storage.googleapis.com/kaggle-media/learn/images/2MA4iMV.png

                                    *A dense layer of two linear units receiving two inputs and a bias.*

上图是由两个线性单元组成的一个密集层或者说全连接层。每个线性单元又由两个input（x0，x1）和一个bias（1）组成。

我们可以认为在神经网络里的每个层都进行某种相关连的简单的转换。通过深度的增加（层数增多）可以以越来越复杂的方式转换输入。

让我们更深入的理解一下Dense Layer：它是神经网络中的一种基本层类型，特点是每一个神经元都与前一层的所有神经元相连接。这意味着每个输入都将影响每个输出。用法：每个神经元接受前一层的所有输出作为输入，并通过weight和bias来计算加权和，最后通过激活函数进行非线性变换。dense layer 的输出可以表示为y=f(Wx+b)，其中f 是激活函数。可以进行复杂的线性变换和非线性变换。

### **The Activation Function**

**什么是激活函数？**

定义：激活函数是神经网络中用来引用非线性特性的函数。

**为什么要使用激活函数？**

如果没有激活函数，神经网络的所有层之间只是线性变换的叠加。无论层数如何增加，最终的输出还是输入的线性组合，无法处理复杂的非线性问题。激活函数引入了非线性，增强了表达能力。

常见的激活函数：

**Sigmoid 函数**：输出值在0到1之间，常用于二分类问题的输出层。输出值可以看作是概率。

**Tanh 函数**：输出值在-1到1之间，常用于隐藏层。

**ReLU（Rectified Linear Unit）函数**：输出值为输入值和0的较大者，计算简单且收敛快，广泛应用于隐藏层。整流函数（Rectifier Function）是激活函数的一种。它将输入的负值截断为零，而正值保持不变，即 ReLU 函数。

**特点**：

- **稀疏激活**：ReLU 函数在输入值为负时输出为零，使得一部分神经元不被激活，从而提高模型的稀疏性和效率。
- **计算简单**：ReLU 计算非常简单，只需比较和选择操作，计算速度快。
- **梯度消失问题**：由于 ReLU 在正值范围内的梯度始终为1，能有效缓解传统激活函数（如 sigmoid 和 tanh）的梯度消失问题。

**Leaky ReLU 函数**：ReLU 的改进版，允许负值有小的输出，减轻“神经元死亡”问题。

**Softmax 函数**：多分类问题的输出层使用，输出为概率分布。输出值的总和为1

!https://storage.googleapis.com/kaggle-media/learn/images/eFry7Yu.png

                                                                   *A rectified linear unit.*

### **Stacking Dense Layers**

!https://storage.googleapis.com/kaggle-media/learn/images/Y5iwFQZ.png

                                  *A stack of dense layers makes a "fully-connected" network.*

上图是一个全连接的堆叠密集层。Stacking Dense Layers指的是将多个密集层（dense）一个接一个的堆叠在一起，形成一个更深的神经网络，每一层的输出作为下一层的输入。前几层可能扑捉到较低级别的特征，而后几层能扑捉到更高级，抽象的特征。为了能够学习复杂的非线性特征，所以在密集层之间引入激活函数。

上图的输出层是一个没有激活函数的线性层。这种方法适用于回归任务，对于预测连续值的任务有很好的表现。如果是分类任务则不适用，分类任务在输出层通常使用使用Softmax或Sigmoid激活函数来输出类别概率。

如果想建立一个上图一样的模型，可以应用下图的代码：

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
# the hidden ReLU layers
layers.Dense(units=4, activation='relu', input_shape=[2]),
layers.Dense(units=3, activation='relu'),
# the linear output layer
layers.Dense(units=1),
])
```

https://www.kaggle.com/code/ryanholbrook/deep-neural-networks

# Chapter3 **Stochastic Gradient Descent（随机梯度下降）**

之前我们创建的模型还不知道任何事情，因为权重是随机设置的。我们需要知道如何训练一个神经网络，还需要知道神经网络是如何学习的。

所有的机器学习任务，都是开始于一组训练数据。每组数据都包括一些特征（inputs）和一个预期结果（output）。训练神经网络意味着调整权重（weight），使它可以transform the features into the target. 如果我们能够成功地训练网络做到这一点，其权重必须以某种方式代表这些特征与训练数据中表达的目标之间的关系。那么如何量化训练数据的好坏呢？1.损失函数（loss function），用来衡量神经网络预测的好坏 。2.优化器（optimizer），可以告诉神经网络如何更改权重。下面来详细介绍一下这两点。

### **The Loss Function**

我们已经看到了如何设计网络的架构，但是我们没有告诉网络要解决什么问题。这就是损失函数的工作。

损失函数是衡量目标的真实值和模型预测值之间的差异。不同的问题需要不同的损失函数。比如说，当我们的任务是预测一些数值类型（房价，或者是汽车的燃油效率等）的目标时，我们都归结为在处理回归（regression）问题。

在回归问题中，常用的损失函数时MAE（mean absolute error 平均绝对误差）。对于每个预测值，MAE通过计算预测值和真实目标的绝对差异来测量差异。

除了MAE，其他常见的回归问题的损失函数还有均方误差（MSE）和Huber损失。

在训练期间，模型使用损失函数使得模型找到正确的权重（损失越低越好）。

### The Optimizer - Stochastic Gradient Descent

我们可以描述一个想让神经网络去解决的问题，但是我们必须告诉它如何去解决它。这就是optimizer。optimizer是一个用来调整权重去最小化损失的一种算法。

实际上在深度学习中所有的最优算法都属于一个家族—— **stochastic gradient descent（随机梯度下降），本质是迭代算法。**

1.准备一些训练数据，然后通过神经网络做一些预测。

2.衡量预测值和真实值之间的损失。

3.最后调整权重，让损失更小。

然后一直重复做这件事直到损失足够小。

!https://storage.googleapis.com/kaggle-media/learn/images/rFI1tIk.gif

                                 *Training a neural network with Stochastic Gradient Descent.*

每一次迭代的数据样本叫做minibatch，而完整的一次训练叫做epoch。如果说我训练的epoch的数量是3个，表示网络看到每个训练样本三次。上面的动画展示了之前章节被训练的SGD的线性模型。

浅红色的点表示整个训练集。而实心红点是minibatch。每当SGD看到一个新的minibatch，它就会将权重向着正确的值调整。一次一次的迭代。这条线最后会收敛到最佳拟合，你可以看到随着权重变得接近其真实值，产生的损失也越来越小。

### **Learning Rate and Batch Size**

图一当中的直线每次都进行轻微的调整，而不是一次调整到位。每次调整的大小是由learning rate来决定的。更小的学习率意味着要学习更多次才能够收敛到最佳拟合。在SGD的训练当中，learning rate和the size of minibatchs是影响最大的两个参数。他们的互相作用通常很微妙，而且不总是显而易见的。

对于大多数工作来说，不是必须进行超参数搜索来得到满意的结果。比如我们可以使用Adam算法，Adam是一种SGD算法，具有自适应学习率，使其适用于大多数问题而不需要参数调整。

### **Adding the Loss and Optimizer**

定义了一个模型之后，可以按照下面的方式为模型添加一个损失函数和优化器：

```
model.compile(
    optimizer="adam",
    loss="mae",
)
```

当然如果你想调整参数，我们也可以使用Keras API。

**PS：为什么是SGD（随机梯度下降）**

梯度是一个向量，告诉我们权重应该如何调整。更精确一点说，它告诉我们如何调整权重让损失变化的最快。我们称这个变化的过程叫梯度下降，因为它使用梯度沿着损失曲线下降到最小值。**Stochastic**意味着由随机决定，因为minibatch是从数据集中随机抽取的样本。以上就是这种方式为什么被称作随机梯度下降。

https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent

# Chapter4 **Overfitting and Underfitting（过拟合和欠拟合）**

现在我们来看一下如何解释这些学习曲线和我们如何使用他们去指导模型发展。尤其是我们会发现underfitting和overfitting，并寻找一些策略去纠正它。

### **Interpreting the Learning Curves**

在训练数据中包括的信息有两种：signal and noise（信号和噪声）。信号是那些可以泛化的，可以帮助模型对新数据进行预测的部分。噪声是那些仅对训练数据有效的部分，是所有来自现实世界的随机波动，或者是那些非信息性，不能真正帮助模型做预测的部分。有些噪声看起来可能很有用，但实际上可能并不是。

我们通过选择在训练集上拥有最小损失的权重或者参数来训练模型。然而为了准确评价模型的表现，我们需要在新的数据集（验证集）上进行评估。

当我们训练模型时，我们会绘制每个周期（epoch）的损失曲线。除此之外我们还会绘制验证集上的损失曲线。我们将这些曲线称之为学习曲线（learning curves）

!https://storage.googleapis.com/kaggle-media/learn/images/tHiVFnM.png

                    *The validation loss gives an estimate of the expected error on unseen data.*

在这个曲线中，训练损失下降的原因可能是模型学习到了信号，也可能是学习到了噪声。但是验证损失只有在学习到了信号时才会下降（因为从数据集学到的噪声不会泛化到新数据上）。

因此，当模型学习到信号时，两个曲线都下降，但当学习到噪声时，两条曲线之间会出现一个gap，gap的大小告诉了我们模型学到的噪音的多少。

理想情况下，我们希望模型学习到的全部是信号，没有噪声。（几乎不可能），所以我们会做出权衡。我们可以让模型学习到更多的信号，同事也学习到了更多噪声。只要这种权衡对我们有利，验证损失就会持续减少。在某个点之后，代价超过了收益，验证损失开始上升。

!https://storage.googleapis.com/kaggle-media/learn/images/eUF6mfo.png

                                                       *Underfitting and overfitting.*

这种权衡表明，我们在训练模型时会出现的两个问题：信号不足或噪声过多。Underfitting（训练不足）是指模型没有学习到足够的信号，导致损失没有达到最低。overfitting是指模型学习到了过多的噪音，导致损失没有达到最低。训练深度学习模型的关键是找到这两者的最佳平衡点。

下面我们将探索几种方法，从训练数据中获取更多的信号，同时减少噪音。

### **Capacity**

模型的容量指的是它能够学习的模式的大小和复杂性。对神经网络来说，很大程度上取决于它有多少神经元以及这些神经元是如何连接在一起的。如果发现你的网络欠拟合了，应该尝试增加容量。

两种方式：1.加宽网络：增加现有层的神经元数量。2.加深网络：增加更多的层。加宽网络更容易学习线性关系，加深网络有助于学习非线性关系。

**Early Stopping**

!https://storage.googleapis.com/kaggle-media/learn/images/eP0gppr.png

                    *We keep the model where the validation loss is at a minimum.*

我们提到，当模型过于热衷于学习噪音时，验证损失在训练过程中可能会开始增加。为了防止这种情况，我们可以在验证损失不再下降时停止训练。这种中断训练的方法称为**早停法**。一旦我们检测到验证损失开始再次上升，就可以将权重重置到验证损失最小的那一点。这样可以确保模型不会继续学习噪音并过拟合数据。除了防止由于训练时间过长而导致的过拟合外，早停法还可以防止由于训练时间不足而导致的欠拟合。只需将训练的 epochs 设置为一个较大的数值（超过实际需要的），早停法会处理其余的工作。

下面是如何用代码实现：

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001,# minimium amount of change to count as an improvement
    patience=20,# how many epochs to wait before stopping
    restore_best_weights=True,
)
```

这些参数表示，如果前20个epochs中，验证损失没有至少提高0.001，那么停止训练并保留找到的最佳模型。

# Chapter5 **Dropout and Batch Normalization（丢弃层和批量归一化）**

之前聊到了神经网络的层的概念。不仅仅是dense layers（密集层）其实在模型中可以添加多种类的层。有些可以像dense layers一样定义在神经元之间的连接，还有一些可以进行预处理或其他形式的转换。（如何理解preprocessing和transformations？

ps：在机器学习和深度学习中，**预处理**和**转换**是两个关键步骤，用于准备数据以便模型能够有效地学习和进行预测。这些步骤可以显著提高模型的性能和稳定性。比如预处理阶段，1.进行数据清洗，处理缺失值和异常值，2.数据标准化和归一化，数据缩放到特定范围，3.特征提取和特征选择。转换阶段：**转换**是指对数据进行的一些数学或逻辑操作，以便更好地适应模型的需求。比如：1.非线性变换，2.编码分类变量，3.降维。关于这两个部分，之后展开）

现在我们来看两种特殊的层，他们本身不包含任何神经单元本身，但是可以为模型增加更多的功能，在各种情况下可能对模型有益处，这两种层在现代架构中被广泛使用。

### **Dropout**

首先是Dropout layer，它可以帮助纠正过拟合。

之前我们讨论过过拟合是由于神经网络在训练数据中学习到噪声（虚假模式）而导致的。

这种虚假模式实际上非常脆弱，因为它依赖于非常特殊的权重组合，往往移除一个权重，这种模式就会被打破。这种想法就是dropout（丢弃）。我们在每一步训练中都随机丢弃一部分层的学习单元，让网络更难在训练数据中学习到虚假模式，让网络去寻找更广泛，普遍的模式，让权重组合变得更稳健。

!https://storage.googleapis.com/kaggle-media/learn/images/a86utxY.gif

              *Here, 50% dropout has been added between the two hidden layers.*

也可以理解为，这种方式创造了更小的集合，预测不再由一个大网络完成，而是由多个小网络组成的集合来完成，小的组合也会犯很多的错误，但同时也会有正确的部分，使得这个集合比任何一个个体都要好。（如果你熟悉随机森林作为决策树的集合，那是相同的理念。ps：补坑，《随机森林决策的理念》）

添加Dropout的代码如下：

```
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.Dense(16),
    # ...
])
```

### **Batch Normalization**

下一个特殊层我们叫做“batch normalization”（批量归一化），它可以帮助纠正训练过程中的缓慢或者不稳定问题。

对神经网络来说，将所有的数据让在同一个尺度上是一个好主意。原因在于SGD会根据数据产生的激活大小去调整权重。那些产生不同大小的激活特征会导致训练行为的不稳定。

如果说输入网络之前，对数据进行归一化是好的行为，那么在网络内部也进行归一化可能会更好。事实上，我们也可以使用一种特殊层来做归一化，这种层就是**batch normalization layer。**它会对每一个batch进行归一化，首先使用自身的均值和标准差进行归一化，然后使用两个可训练的重新缩放的参数将数据放到一个新尺度上。实际上，批量归一化是对输入进行了一种协调的重新缩放。

通常情况下，批量归一化是作为优化过程的辅助而被添加的（虽然有时它也可能帮助提高预测表现）。带有批量归一化的模型通常需要更少的epochs来完成训练。此外，批量归一化也可以修复一些造成训练“stuck”的问题。如果你在训练中遇到困难，考虑在模型中添加批量归一化是一个好主意。

### **Adding Batch Normalization**

批量归一化通常可以放在一个网络的任何一个位置，比如可以放在一个层之后

```
layers.Dense(16, activation='relu'),
layers.BatchNormalization(),
```

或者放在一个层和它的激活函数之间

```
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),
```

如果我们把它加到神经网络，并且作为第一层，就可以当作是一种自适应处理器，类似于

Sci-Kit Learn的`StandardScaler`。批量归一化应用在复杂的数据集上会非常有效。

# **Chapter6 Binary Classification（二分类）**

之前我们学习过了神经网络如何处理回归问题。现在我们学习如何处理分类问题。和处理回归问题时学到的大部分东西都差不多，主要的区别在于使用到的损失函数以及最后一层（输出的种类不同）。

分类问题是常见的机器学习问题之一。比如预测一个顾客是否会订阅会员，或者信用卡是否被盗刷，或者判断一封邮件是否是垃圾邮件。这些问题都是分类问题。在我们的原生数据中，这个分类可能是一个字符串，比如YES 或者 NO，或者是dog或cat。在使用这些数据之前，我们将分配一个类标签：一个类为0，另一个为1。分配数字标签将数据置于神经网络可以使用的形式。

### **Accuracy and Cross-Entropy**

准确性（**Accuracy**）是衡量分类问题成功与否的一个因素，准确性是所有预测和成功预测的比例——成功预测/所有预测。如果所有的预测都正确，那么得分为1.0。

准确性的问题是无法作为损失函数去应用。比如SGD需要的是一个变化平滑的损失函数，但是准确性是一个比例，变化是跳跃式的。所以我们必须选择一个替代者去承担损失函数的任务，这个替代者就是 ***cross-entropy* function（）交叉熵。**

回想一下在回归任务中，我们的目标是最小化期待值和预测值之间的差。我们选择MAE去衡量这之间的差距。对于分类，我们想要的是概率之间的距离，这就是交叉熵提供的。交叉熵是一种从一个概率分布到另一个概率分布距离的度量。

!https://storage.googleapis.com/kaggle-media/learn/images/DwVV9bR.png

                                     *Cross-entropy penalizes incorrect probability predictions.*

我们希望我们的网络以1.0的概率预测正确的类。预测概率离1.0越远，交叉熵损失就越大。

使用交叉熵去处理分类损失，其他的指标（比如准确性Accuracy）也将会提升。

### **Making Probabilities with the Sigmoid Function**

交叉熵和准确率函数都需要概率作为输入，也就是一个0～1之间的数字。通过一个全连接层将真实的输出值转化为概率，Sigmoid Function就可以实现这个功能。

!https://storage.googleapis.com/kaggle-media/learn/images/FYbRvJo.png

                                     *The sigmoid function maps real numbers into the interval*

为了得到最后的预测结果，我们定义一个阈值，通常是0.5，低于0.5意味着是标签是0的类，大于0.5意味着是标签时1的类。

**应用场景**：

1.二分类问题，通常在最后一层使用，将正常的值转换为概率

2.损失函数：交叉熵损失函数要求输入为概率，所以需要将模型输出通过Sigmoid函数转换为概率。

- DeepLearning书单：

[Deep Learning](https://www.deeplearningbook.org/)

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
