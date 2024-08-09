# Brain Tumor MRI Classification: TensorFlow CNN

补了一些机器学习和深度学习的基础知识，开始直接实践上手项目。

开始尝试的第一个项目（来自Kaggle：脑肿瘤MRI分类）。

### **Introduction**

这个项目，使用 Brain Tumor 数据集，使用CNN去进行图像分类。因为这个数据集很小，如果我们训练一个神经网络，不会带来很好的结果。因此，我们将使用Transfer Learning的概念去训练模型得到更精确的结果。

**Importing Libraries**

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings
for dirname, _, filenamesin os.walk('/kaggle/input'):
    for filenamein filenames:
        print(os.path.join(dirname, filename))
```

使用了matplotlib，numpy，pandas，seaborn，cv2，tensorflow，sklearn

**Color**

```python
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

sns.palplot(colors_dark)
sns.palplot(colors_green)
sns.palplot(colors_red)
```

![Untitled](Brain%20Tumor%20MRI%20Classification%20TensorFlow%20CNN%20bae18f36f1c94a62bfc588d600af6823/Untitled.png)

### **Data Preperation**

`labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']`

我们首先将目录中的所有图像附加到Python列表中，然后在调整大小后将它们转换为numpy数组。

```python
//声明三个变量，训练集X，训练集y，size
X_train = []   //存储图像
y_train = []   //存储标签 
image_size = 150   //图像大小150
//循环遍历每个标签
for i in labels:
    //生成训练数据的文件夹路径，例如../input/brain-tumor-classification-mri/Training/tumor
    folderPath = os.path.join('../input/brain-tumor-classification-mri','Training',i) 
    //循环遍历该文件夹下的所有文件。 tqdm显示进度条
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j)) //读图像文件‘j'，储存为img
        img = cv2.resize(img,(image_size, image_size)) //将图片size调整成150x150
        X_train.append(img) //图像添加到X train的列表中
        y_train.append(i) //对应的标签添加到 y train的列表中
//循环遍历每个标签，对测试集进行一样的处理，需要注意的是，本段代码将测试集也一样加入到了训练数据中
for i in labels:
    folderPath = os.path.join('../input/brain-tumor-classification-mri','Testing',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)

//将X_train和y_train列表转换为NumPy数组。
X_train = np.array(X_train)
y_train = np.array(y_train)
```

![Untitled](Brain%20Tumor%20MRI%20Classification%20TensorFlow%20CNN%20bae18f36f1c94a62bfc588d600af6823/Untitled%201.png)

```python
X_train, y_train = shuffle(X_train,y_train, random_state=101) //使用sklearn.utils库的shuffle函数，对输入的数组进行洗牌
//random_state=101 保证洗牌的随机性是可重复的，每次运行代码会产生相同的结果
X_train.shapeX_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
```

```python
//分割训练集和测试集。train_test_split来自sklearn库。参数分别是1.图像数据 2.标签 3.表示10%的数据用于测试集 4.随机分割可重复
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.1,random_state=101)
//使用热编码将标签转换为数值
y_train_new = []  //初始化一个空列表，存储训练标签的索引
//循环遍历训练标签
for i in y_train:
    y_train_new.append(labels.index(i)) //将标签i转换为在labels列表中的索引，添加到y train new中
y_train = y_train_new //将原始的训练标签替换为索引形式的标签
y_train = tf.keras.utils.to_categorical(y_train) //将标签转换为one-hot编码形式

//对测试集做同样的处理
y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test = y_test_new
y_test = tf.keras.utils.to_categorical(y_test)
```

**为什么需要将原始的训练标签替换为索引？** 

计算机处理分类任务时更容易操作数值类型而不是字符串，将标签名称转换为索引（数值形式）后，可以更高效的进行计算和存储。

One-hot编码是一种向量表示法，其中只有一个元素是1，其他都是0 。比如标签['cat', 'dog', 'mouse']，它们的索引是[0, 1, 2]

对应的One-hot编码是

- 'cat' -> [1, 0, 0]
- 'dog' -> [0, 1, 0]
- 'mouse' -> [0, 0, 1]

### **Transfer Learning**

深度卷积神经网络模型在训练非常大的数据集的时候，可能会花费几天甚至几周的时间。有一个简化这个过程的方式就是重用预训练模型的权重，这些模型是为标准的计算机视觉基准数据集（例如ImageNet图像识别任务）开发的。可以直接下载并使用顶级性能的模型，或者集成到自己的新模型中，来解决自己的问题。

在这我使用**EfficientNetB0模型。**权重来自于ImageNet数据集。

```python
effnet = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))
```

include_top设置成False可以让预训练模型的网络不包括第一层或者最后一层，允许我们自己按照自己的情况添加output层。

```python
model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)
model.summary()  //打印出模型的结构，各层的输出形状，参数数量等信息
```

**GlobalAveragePooling2D：**这个层和CNNs中的最大池化层相似，唯一的不同是它池化的时候使用平均值代替最大值。这样可以减少训练时的计算负载。

**Dropout：**该层省略了该层每一步的一些神经元，使神经元与神经元更加独立。对避免过拟合有帮助。被省略的神经元是随机选取的。rate参数表示神经元激活被设置为0的概率。

**Dense：**这个输出层实现了4种可能性的图片分类。使用softmax函数作为激活函数。

```python
model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])
```

配置模型，指定损失函数，优化器和评估指标。

categorical_crossentropy是一个常用的损失函数，适用于多分类问题。如果标签是独热编码的时候，使用这个损失函数。它会计算概率分布和真实标签之间的差异，并引导模型的参数朝着减少这种差异的方向更新。

optimizer = 'Adam'：Adam是一种流行的优化算法，它结合了两种常用优化方法（RMSProp 和 SGD with momentum）的优点，有自适应学习率，有助于加快收敛速度并避免局部最小值。

metrics= ['accuracy']：accuracy表示模型预测正确的样本比例（多分类问题中）。

**Callbacks（回调函数）：**可以帮我更快的修复错误，构建更好的模型。可以可视化模型的训练进展，甚至通过实现Early Stopping 或者在每次迭代中自定义学习率来防止overfitting。

回调函数是一组在训练过程应用的函数，可以查看模型的内部状态和统计信息。

 I'll be using **TensorBoard, ModelCheckpoint and ReduceLROnPlateau** callback functions

```python
tensorboard = TensorBoard(log_dir = 'logs')
checkpoint = ModelCheckpoint("effnet.h5",monitor="val_accuracy",save_best_only=True,mode="auto",verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)
```

TensorBoard(log_dir = 'logs')：TensorBoard用来可视化训练过程中的指标（比如损失和准确率等）和模型结构。logs是指定文件的保存位置。

ModelCheckpoint：用于在训练期间保存模型的权重。"effnet.h5"：保存模型的文件名。monitor="val_accuracy"：监控验证集的准确率。save_best_only=True：只保存性能最好的模型。mode="auto"：自动选择监控指标的最佳模式,verbose=1：在每次保存模型时输出日志信息。

ReduceLROnPlateau：用于在模型性能停滞时自动降低学习率，从而帮助模型更好的收敛。

### **Training The Model**

Note：在本地环境只是用CPU训练，花费2小时。如果使用GPU大概需要5分钟。

```python
history = model.fit(X_train,y_train,validation_split=0.1, epochs =12, verbose=1, batch_size=32,
                   callbacks=[tensorboard,checkpoint,reduce_lr])
```

这行代码👆：通过指定数据集，训练参数和回调函数来启动模型的训练过程。模型将被训练12个epoch，每个epoch中使用32个样本的批次进行梯度更新。通过validation_split参数，模型将在每个epoch结束的时候评估其在10%的验证数据上的性能。回调函数将协助监控训练进度，保存最佳模型权重以及在需要时调整学习率。训练的历史记录会保存在history变量中。

> 
> 
> 
> ```
> Epoch 1/12
> 83/83 [==============================] - 25s 157ms/step - loss: 0.6583 - accuracy: 0.7582 - val_loss: 0.6261 - val_accuracy: 0.8061
> 
> Epoch 00001: val_accuracy improved from -inf to 0.80612, saving model to effnet.h5
> Epoch 2/12
> 83/83 [==============================] - 9s 114ms/step - loss: 0.1961 - accuracy: 0.9336 - val_loss: 0.3032 - val_accuracy: 0.9048
> 
> Epoch 00002: val_accuracy improved from 0.80612 to 0.90476, saving model to effnet.h5
> Epoch 3/12
> 83/83 [==============================] - 10s 115ms/step - loss: 0.1268 - accuracy: 0.9585 - val_loss: 0.3685 - val_accuracy: 0.8673
> 
> Epoch 00003: val_accuracy did not improve from 0.90476
> Epoch 4/12
> 83/83 [==============================] - 9s 114ms/step - loss: 0.1208 - accuracy: 0.9580 - val_loss: 0.8419 - val_accuracy: 0.7857
> 
> Epoch 00004: val_accuracy did not improve from 0.90476
> 
> Epoch 00004: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
> Epoch 5/12
> 83/83 [==============================] - 10s 117ms/step - loss: 0.0631 - accuracy: 0.9824 - val_loss: 0.2406 - val_accuracy: 0.9150
> 
> Epoch 00005: val_accuracy improved from 0.90476 to 0.91497, saving model to effnet.h5
> Epoch 6/12
> 83/83 [==============================] - 10s 116ms/step - loss: 0.0243 - accuracy: 0.9917 - val_loss: 0.1829 - val_accuracy: 0.9354
> 
> Epoch 00006: val_accuracy improved from 0.91497 to 0.93537, saving model to effnet.h5
> Epoch 7/12
> 83/83 [==============================] - 10s 117ms/step - loss: 0.0122 - accuracy: 0.9979 - val_loss: 0.1259 - val_accuracy: 0.9694
> 
> Epoch 00007: val_accuracy improved from 0.93537 to 0.96939, saving model to effnet.h5
> Epoch 8/12
> 83/83 [==============================] - 10s 117ms/step - loss: 0.0110 - accuracy: 0.9978 - val_loss: 0.1521 - val_accuracy: 0.9592
> 
> Epoch 00008: val_accuracy did not improve from 0.96939
> Epoch 9/12
> 83/83 [==============================] - 10s 116ms/step - loss: 0.0147 - accuracy: 0.9955 - val_loss: 0.1117 - val_accuracy: 0.9728
> 
> Epoch 00009: val_accuracy improved from 0.96939 to 0.97279, saving model to effnet.h5
> Epoch 10/12
> 83/83 [==============================] - 10s 118ms/step - loss: 0.0085 - accuracy: 0.9981 - val_loss: 0.1045 - val_accuracy: 0.9524
> 
> Epoch 00010: val_accuracy did not improve from 0.97279
> Epoch 11/12
> 83/83 [==============================] - 10s 117ms/step - loss: 0.0070 - accuracy: 0.9975 - val_loss: 0.0815 - val_accuracy: 0.9728
> 
> Epoch 00011: val_accuracy did not improve from 0.97279
> 
> Epoch 00011: ReduceLROnPlateau reducing learning rate to 9.000000427477062e-05.
> Epoch 12/12
> 83/83 [==============================] - 10s 116ms/step - loss: 0.0059 - accuracy: 0.9993 - val_loss: 0.0785 - val_accuracy: 0.9728
> 
> Epoch 00012: val_accuracy did not improve from 0.97279
> ```
> 

![Untitled](Brain%20Tumor%20MRI%20Classification%20TensorFlow%20CNN%20bae18f36f1c94a62bfc588d600af6823/Untitled%202.png)

### **Prediction**

```
pred = model.predict(X_test) //使用训练好的模型对测试数据集 X_test进行预测
pred = np.argmax(pred,axis=1) //pred的每一行包含所有类别的预测概率
y_test_new = np.argmax(y_test,axis=1) //将真实标签y_test转换为可比较的形式
```

predict函数返回的是每个样本对应的概率分布。

pred = np.argmax(pred,axis=1) ：pred的每一行包含所有类别的预测概率，argmax函数用于找到每行预测数组中的最大值的索引。axis=1，最终结果是一个一维数组。

y_test也是采用了独热编码（one-hot encoding）的格式，其中每一行对应一个二进制向量，表示真实标签，对y_test应用argmax可以将这些独热编码的向量转换回它们原始的类别索引，使之更容易与预测标签进行比较。结果y_test_new是一个一维数组，其中包含每个测试样本的真实类别索引

通过将预测概率和真实标签都转化为类别索引，可以直接比较他们，从而评估模型的性能，比如准确率等。

### **Evaluation**

```python
print(classification_report(y_test_new,pred)) 
//classification_report函数来自sklearn.metrics模块的一个函数生成并打印分类模型的性能报告
```

![Untitled](Brain%20Tumor%20MRI%20Classification%20TensorFlow%20CNN%20bae18f36f1c94a62bfc588d600af6823/Untitled%203.png)

Precision：`Precision = TP / (TP + FP)`，其中 TP 是真正例（True Positives），FP 是假正例（False Positives）。

Recall（召回率）：`Recall = TP / (TP + FN)`，其中 TP 是真正例，FN 是假负例（False Negatives）。

F1-score：`F1 = 2 * (Precision * Recall) / (Precision + Recall)`

Support（支持度）：每个类别在测试集中出现的样本数量。

accuracy：所有样本的总体准确率

macro avg：宏平均，对于所有类别的精确率，召回率和F1分数进行简单平均

weighted avg：加权平均，召回率和F1分数进行加权平均。

**PS：补充一下统计知识（简单平均和加权平均）**

简单平均是指对多个数据点取算数平均值。不考虑每个数据点的重要性或者数量。简而言之每个数据的权重都是同等的。

加权平均是指对每个数据点X权重，然后在求和再➗权重之和。加权平均考虑了每个类别在数据集中出现的频率。适合处理类别不平衡的情况。

![Untitled](Brain%20Tumor%20MRI%20Classification%20TensorFlow%20CNN%20bae18f36f1c94a62bfc588d600af6823/Untitled%204.png)

### **Conclusion**

这个项目使用了迁移学习的CNN，得到了98%的准确率。