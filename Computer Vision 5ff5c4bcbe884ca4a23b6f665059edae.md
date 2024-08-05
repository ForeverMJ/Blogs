# Computer Vision

[Intro to Deep Learnling (1)](https://www.notion.so/Intro-to-Deep-Learnling-1-a705fa043a29467482232311095ead37?pvs=21)

⬆️基础知识：点击链接跳转

### KEY WORDS

**the base of CV**

**CNN卷积神经网络**

**filter，detect，pooling**

**The Sliding Window**

**Data Augmentation**

# Chapter1 **The Convolutional Classifier**

### Intro

这个课程会介绍关于计算机视觉的基础概念。我们的目标是学习如何让神经网络可以理解一个图片去解决人类视觉系统可以解决的一些问题。

卷积神经网络（CNN）在机器视觉方面表现最好。**介绍一下卷积（Convolutional）**：

在后面，会详细解释为什么这个结构在解决计算机视觉问题方面非常有效。

我们将会主要学习如何处理图片分类（**image classification**）问题。比如如何识别图片中是一只狗或者小猫。之后我们还会学习更高级的东西，比如 generative adversarial networks and image segmentation.【生成式对抗网络（GAN）和图像分割】

### **The Convolutional Classifier**

一个用于图像分类的卷积神经网络（convnet）由两个部分组成：卷积基（convolutional base）和全连接层头（dense head）。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled.png)

Base是用来从图像中提取特征的。它主要由执行卷积操作的层组成，但通常也包括其他类型的层。

（后面会详细解释）。Head通常用来分类。主要由dense layers组成，但是也可能包括另一些层，比如dropout层。

我们怎么衡量一个视觉特征呢？一个特征可能是一条线，一种颜色，一种感觉，一种形状，一个关系，或者一些复杂的结合体。

### **Training the Classifier**

通过训练，网络的目标是学习两件事情：

1.从一张图片中提取什么特征（base）

2.哪个类别与哪些特征相关（全连接层头）

现如今，卷积网络很少被从头开始训练。常见的是，我们重用一个预训练模型作为base。然后连接一个未训练的head。因为head通常只有一些全连接层（dense layers），非常精确的分类可以通过很少量的相关数据训练出来。

这种重复使用一个预训练模型的技术叫做transfer learning（转移学习），因为它很有效，所以大部分图像分类器都会使用它。

### **Example - Train a Convnet Classifier**

**Step 1 - Load Data**

**Step 2 - Define Pretrained Base**

**Step 3 - Attach Head**

**Step 4 - Train**

### **Conclusion**

（后补）

# Chapter2 Convolution and ReLU

### **Feature Extraction**

让我们讨论一下在网络当中，这些层的作用是什么。我们可以看到convolution，ReLU，maximum pooling这三个运算是如何应用在特称提取过程中的。

特征提取包括三个基本运算：

1.通过一个特殊的特征去Filter一个图像（convolution）

2.在删选过的图像中检测该特征（ReLU）

3.Condense浓缩图像使其加强特征（maximum pooling）

下面的图片解释了这个过程，这三个运算是如何应用在一张原生图片中的。

![https://storage.googleapis.com/kaggle-media/learn/images/IYO9lqp.png](https://storage.googleapis.com/kaggle-media/learn/images/IYO9lqp.png)

                                                *The three steps of feature extraction.*

通常，网络将在单个图像上并行执行多次提取。在现代convnets中，Base的最后一层产生1000多个独特的视觉功能并不罕见。

### **Filter with Convolution**

卷积层执行过滤步骤。你可以使用Keras来定义一个卷积层（convolutional layer）

```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3),# activation is None# More layers follow])
```

我们可以通过寻找weights和activations之间的关系来理解这些参数。

### **Weights**

一个卷积网络在训练中学习到的weights主要包含在卷积层中。这些权重我们称作是kernels（卷积核）。我们可以用数组来表示。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%201.png)

卷积核通过扫描图像并生成像素值的加权和来工作。卷积核就像一个偏光镜，强调或减弱一些信息模式。

![https://storage.googleapis.com/kaggle-media/learn/images/j3lk26U.png](https://storage.googleapis.com/kaggle-media/learn/images/j3lk26U.png)

                                                             *A kernel acts as a kind of lens.*

卷积核定义了卷积层如何连接到后面的层。卷积核将Output中的每个神经元连接了Input中的九个神经元。通过设定卷积核的维度（kernel_size），你可以告诉卷积神经网络如何形成这些连接。大多数时候，卷积核的维度都是奇数—比如`kernel_size=(3, 3)` or `(5, 5)`。这样单个像素位于中间。但这不是必须的。

**卷积层中的卷积核决定了它创建的特征类型。在训练过程中，卷积神经网络会尝试学习解决分类问题所需的特征，这意味着找到卷积核的最佳值。**

### **Activations**

CNN中的激活函数叫做feature maps（特征图）。它是对一个图片进行filter之后产生的结果。它包括卷积核提取出来的视觉特征。下面是一些卷积核生成的特征图。

![https://storage.googleapis.com/kaggle-media/learn/images/JxBwchH.png](https://storage.googleapis.com/kaggle-media/learn/images/JxBwchH.png)

                                                                 *Kernels and features.*

从卷积核中的数字模式中，你可以看出它生成的特征图类型。通常来说，卷积将在输入强调的内容和卷积核中的正数形状相匹配。左图和中间的图会过滤出水平形状的图。

通过filters参数，可以告诉卷积层你想输出多少个特征图作为输出。

### **Detect with ReLU**

过滤之后，特征图会通过激活函数。The rectifiter function（整流函数） has a graph like this：

![https://storage.googleapis.com/kaggle-media/learn/images/DxGJuTH.png](https://storage.googleapis.com/kaggle-media/learn/images/DxGJuTH.png)

*The graph of the rectifier function looks like a line with the negative part "rectified" to 0.*

带着一个整流器的神经元叫做 rectified linear unit（整形线性单元）。也称作ReLU activation或者ReLu function。

ReLu函数可以在专属的激活层中定义。但大部分情况下我们会直接应用一个叫Conv2D的一个激活函数。

```
model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3, activation='relu')
# More layers follow])
```

你可以认为激活函数根据某种重要性标准对像素值进行评分。ReLu activation表示了负值不重要，将他们设置成0（"Everything unimportant is equally unimportant.”每个不重要的东西都是一样不重要）。

像其他激活函数一样，ReLU函数是非线性的。这意味着网络中所有层的总效果与仅通过叠加效果得到的结果不同——这与你仅使用单一层可以达到的效果相同。非线性确保了特征在深入网络时会以有趣的方式组合在一起。（我们将在之后进一步探讨这种“特征复合”（feature compounding）。）

### **Conclusion**

我们学到了卷积神经网络进行特征提取的两个步骤：通过Conv2D层进行过滤（filter），然后通过relu激活函数进行detect（检测）。

# Chapter3 **Maximum Pooling**

我们在来看第三个：condense with maximum pooling（通过最大池化层缩减和提取特征），在Keras通过MaxPool2D layer来完成。

### **Condense with Maximum Pooling**

```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64, kernel_size=3),# activation is Nonelayers.MaxPool2D(pool_size=2),
# More layers follon shiw])
```

MaxPool2D层非常像Conv2D层，但是不同点在于MaxPool2D采用了一个简单的最大值函数代替卷积核（Kernel）。pool Size 参数类似 kernel size。一个MaxPool2D层没有任何可训练的权重。

我们需要记住MaxPool2D是一个缩减和提取的步骤。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%202.png)

我们会注意到在应用ReLUfunction（Detect）后，特征图最终会有很多“dead space”，意味着大片区域只包括0‘s（图片中的黑色区域）。对于整个网络来说增加了模型的规模，而且没有增加有用的信息。所以我们倾向于浓缩特征图来保持大部分有用的部分（特征本身）。

这实际上就是maximum pooling（最大池化）的作用。最大池化通过将原生特征图中的激活值替换为最大值。有效的浓缩了特征图，使得活跃像素的比例增加。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%203.png)

### **Example - Apply Maximum Pooling**

（后补）

### Chapter4 The Sliding Window

卷积和池化处理都有一个共同的特征：sliding window （滑动窗口）。卷积的时候，这个窗口是由卷积核的尺寸，也就是kernelsize参数决定的。池化的时候，是pooling window，由poolsize参数决定。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%204.png)

有两个额外的参数也会影响池化和卷积，这两个参数是strides of the window（窗口的步幅），以及是否在图像边缘使用填充（padding）。参数strides表示窗口每一步应该怎么移动，参数padding描述了我们如何处理输入的边缘像素。

使用这两个参数，我们建立一个卷积层，和一个池化层。👇

```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=2,
                     strides=1,
                     padding='same')
# More layers follow])
```

### **Stride**

窗口每一步移动的距离叫做stride。具体说明一下什么是stride：从左到右移动一步然后从上向下移动一步。strides=(2, 2)意味着一次移动两个像素。不同的stride有什么不同的影响呢？如果步幅超过了1，滑动窗口会跳过一些像素。因为我们想使用一些高质量特征去分类，卷积层通常是strides=(1, 1)。增加步幅意味着我们会错过一些有潜力的有价值的信息。最大池化层的stride values通常会超过1，比如`(2, 2)` or `(3, 3)`，但是不会超过窗口本身的大小。步幅在每个方向上都相同，在设置参数时，比如（2，2）只需要设置strides=2就可以了。

### **Padding**

当滑动窗口计算的时候，有一个问题是在边缘时应该做什么。完全保持在输入图像内意味着窗口永远不会像输入中的其他像素那样完全位于这些边界像素上。既然我们没有完全一样地对待所有像素，会有问题吗？

卷积对这些边界值的作用取决于padding参数。在TensorFlow，你可以有两个选择padding=‘same’ 或者 padding=‘valid’。

当我们设置padding=‘valid’的时候，卷积窗口将会完全保持在输入内部。缺点是输出尺寸会缩小（像素值丢失），卷积核越大缩小的程度越大。这会限制一个网络可以包含的层的数量，尤其是输入尺寸很小的时候。

或者设置padding=‘same’。技巧是用0填充边界。填充的 0 数量刚好使输出尺寸与输入尺寸相同。缺点是会降低边缘像素的影响力。

VGG模型在卷积层上使用same padding。很多现代卷积网络都结合两者使用。

Ps:什么是VGG

### **Example - Exploring Sliding Windows**

（后补）

# Chapter5 **Custom Convnets**

现在基本上卷积神经网络用来提取特征的layers都学习完了，下面我们来整合一下，建立一个自己的神经网络。

### **Simple to Refined**

来回顾一下，卷积神经网络提取特征是通过三个方式：filter，detect，and condense。一轮特征提取只能提取一些简单的相关特征，比如简单的线条和对比度。这太简单了以至于无法解决绝大多数的分类问题。所以神经网络会不断重复提取，随着网络深度的增加，特征会变得更复杂。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%205.png)

### **Convolutional Blocks**

通过一个卷积区块组成的长链来提取。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%206.png)

这些卷积区块是由Conv2D和MaxPool2D层组合而成的。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%207.png)

每一个block表示一轮提取，通过组合这些区块，可以结合和从新组合出新的特征，塑造它们来更好的适应特定问题。现代神经网络的深度结构是能够应用这些复杂特征工程的原因。

### **Example - Design a Convnet**

（后补）

# Chapter6 **Data Augmentation**

之前学了卷积分类的基础，下面来看看更高阶的内容。关于一个小技巧，叫做data augmentation（数据增加），可以增强图片分类能力。

### **The Usefulness of Fake Data**

提升机器学习模型表现的最好方式无疑是用更多数据去训练。模型从更多的例子当中学习，模型就会更好的意识到图片关系中的不同。更多的数据帮助模型更好的泛化。

最简单的方式是利用已经有的数据，如果我们可以保持图片的类别不变的方式去转换数据集中的图像，就能教会分类器忽略这些转换。例如，不论汽车在图片中朝左还是朝右，都不会改变它是汽车而不是卡车的事实。因此，如果我们用反转过的图像增强训练数据，分类器就会忽略左右的差异。

这就是数据增强的核心：增加一些看起来像真实数据的虚假数据。

### **Using Data Augmentation**

有许多变换可以被应用到数据增强。可以旋转图片，调整颜色或者对比度。或者其他方法，也可以将这些方式结合起来应用。

![Untitled](Computer%20Vision%205ff5c4bcbe884ca4a23b6f665059edae/Untitled%208.png)

数据增强通常是被在线完成的，这意味着图片在被输入网络进行训练时进行变换。训练通常在数据的小批量上进行的。

每次在训练过程中使用图像，都会应用一个新的随机变换的内容，通过这种方式，模型总是会看到和之前稍有不同的内容。这种训练数据额外的变化，有助于模型在处理新数据时表现的更好。

但是不是每种变换在特定问题上都是有用的。不论我们使用那种变换方式，都不能混淆类别。比如在识别数字时，旋转图像会混淆6和9. 寻找好的数据增强方法的最佳途径和大多数机器学习问题相同：尝试并观察效果。

### **Example - Training with Data Augmentation**

（后补）