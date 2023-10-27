# Introduction to ANN
![](https://files.mdnice.com/user/38974/d121abaa-5d8b-4033-aa2a-99d2a9deda18.png)
## 神经网络初印象

<div align="center">
<img src="https://files.mdnice.com/user/38974/369bf394-c472-4c89-86ea-e903a9641d6c.jpg" alt="猫" width="200" height="200"><img src="https://files.mdnice.com/user/38974/675862a3-e195-4192-826b-8d2a3594fad6.jpg" alt="Tom" width="200"
height="200">
</div>
<div align="center">
<img src="https://files.mdnice.com/user/38974/a2b7a817-1b41-4a7b-9626-e3e96e343b50.jpg" alt="老虎" width="200" height="200"><img src="https://files.mdnice.com/user/38974/a5758faa-673f-4272-8e86-187d57c4f21d.jpg" alt="猫耳娘" width="200" height="200">
</div>


## 经典的人工神经网络
### 人工神经网络的基本概念
### 从人工神经元模型谈起
![](https://files.mdnice.com/user/38974/d6f191ca-202d-4ff8-a637-03d27d93cb28.png)

  设$x=[x1,x2,...,xn]是输入向量，\omega=[\omega_1,\omega_2,...,\omega_n]是权值向量，o是神经元的输出，f是响应函数$。
  
  因此我们有
  $$
      o=f(\Sigma_{i=1}^{n}\omega_ix_i)
  $$
### 响应函数
- 阈值函数
$$
\begin{equation}
f(sum):=\left \{
	\begin{aligned}
	1 \quad sum>0\\
	0 \quad sum<0\\ \nonumber
	\end{aligned}
	\right 
  .
\end{equation}
$$
- Sigmoid函数
$$
f(sum) \triangleq \frac{1}{1+\exp^{-\lambda sum}}
$$
<img src="https://files.mdnice.com/user/38974/bf83809c-bbac-430f-865d-460af176ac09.png" alt="sigmoid" width=600 height=300 >

- Relu函数
$$
\begin{equation}
f(sum)\triangleq\left \{
	\begin{aligned}
	x \quad sum>0\\
	0 \quad sum\leq0\\  \nonumber
	\end{aligned}
	\right 
  .
\end{equation}
$$
<img src="https://files.mdnice.com/user/38974/b48566ea-0e34-4c84-9dc2-fb9a8883f3fe.png" alt="Relu" width="600" heigth="100" >

### 线性分类与非线性分类：一层到两层的转变

### 人工神经元的连接方式
神经网络根据神经元的连接方式分为前向网络和反馈网络。前向网络由输入层的输入得到输出层的响应。反馈网络由输入层的初始输入得到输出层的初始响应，然后输出层的响应作为下一时刻输入层的输入。对于反馈网络最终收敛的平衡状态称为吸引子。
### 单层、多层前向神经网络
<img src="https://files.mdnice.com/user/38974/1772efab-c753-4e53-b62a-549d57d25bd4.png" alt="ANN4" >


<img src="https://files.mdnice.com/user/38974/ecd96169-3063-4c18-8c93-d79ec007d1db.png" alt="ANN3">

对于神经网络的设计，除了神经元的连接结构，更重要的是学习。对于分类或函数拟合的神经网络，是通过一组训练例子的输入与输出之间的映射关系进行学习的。
### 误差函数
首先给所有参数赋上随机值。我们使用这些随机生成的参数值，来预测训练数据中的样本。样本的预测目标(输出向量)为$o_i$，实际目标(给定向量)为$d_i$。那么，我们设置损失函数$loss$，计算公式如下:
$$
loss\triangleq \Sigma(d_i-o_i)^2
$$
我们的目的就是通过优化权重参数使对所有训练数据的损失和尽可能的小。

<img src="https://files.mdnice.com/user/38974/62c360e5-b9d4-4b21-a925-323800e406c7.png">

### 人工神经网络的学习机理
- 学习率$(learning rate，LR)$

  规定了参数修正的幅度。若损失函数要求$\omega_1、\omega_2$变小，则$LR$可调节$\omega_1、\omega_2$是应该减小$0.1、1.0还是10$。

  学习率过大导致神经元失活($Relu$)

- 反向传播：链式法则

### 参数过多问题

### 过拟合问题 
- $overfitting$
  以下是两层各五个神经元拟合出来的随机取点分类
  <div align="center">
  <img src="https://files.mdnice.com/user/38974/d1f9d4be-c9eb-4cc7-9e7a-f89171a10549.png" alt="overfotting" width="200" height="200"><img src="https://files.mdnice.com/user/38974/08e5bdfe-7823-447f-98de-0b6bd359f98a.png" alt="overfotting" width="200" height="200">  
  </div>
- $Solution$ 

  增加训练集量

  减少模型复杂度

  损失函数增加惩罚项
$$
Loss = Loss + \lambda|W|^2 
$$

### A trick——Dropout
  <img src="https://files.mdnice.com/user/38974/6980417e-d78f-4a4b-b261-d2b2180786fd.png" > 

$Dropout$ 适用于全连接层，可以有效缓解过拟合现象以及减轻神经网络负担
- 一些好用神经网络模型中应用$dropout$
<img src="https://files.mdnice.com/user/38974/67556926-b900-476c-a48d-cc57dba5d4f7.jpg">

- 为什么$Dropout$可以解决过拟合？
  
  取平均的作用

  减少神经元之间复杂的共适应关系

  因为$dropout$导致两个神经元不一定每次都在同一个网络中出现，这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况。
### 一些演示   
  [演示1](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html) 
  
  [演示2](http://playground.tensorflow.org)

## 深度神经网络简介
深度学习是一个可以用于对数据进行不同层次抽象的多层神经网络。深度学习的基本思想是设计对应的层次结构，来构建一个深层的神经网络。

对一张图像来说，输人是这张图像的像素值，底层一般可以学习到是否存在某一固定方向的物体边缘，再高层则可以学习到由这些物体边缘组成的简单图案，更高层则可以学习到具体的物体，一般都是由底层学习到的边缘和图案所组成的。深度学习最重要的特点就是这
些中间层的特征不是由人手工选取得到的，而是通过一种学习的方式从数据中获得的。

主要的深度神经网络包括卷积神经网络 $(CNN)$、循环神经网络 $(RNN)$ 和长短期记忆 $(LSTM)$网络等

### 卷积神经网络简介
- CNN和多层神经网络的区别

  深度学习最重要的特点就是这些中间层的特征不是由人手工选取得到的，而是通过一种学习的方式从数据中得到的？CNN的神经元只和前一层的部分神经元结点相连，并且同一层中某些神经元之间的连接权重ω和偏移b共享，这样大大减少了需要训练参数的数量
<img src="https://files.mdnice.com/user/38974/93b27596-fa92-4a2f-ae75-cc4d1090a211.png" alt="CNN1">

- 输入层
  
  在CNN 的输入层中，数据 (图片) 输入的格式与全连接神经网络的输入格式(一维向量)不太一样。CNN 输入层的输入格式保留了图片本身的结构。对于黑白的 28 x28 的图片,CNN的输入是一个28x28 的二维神经元，对于RGB格式的28x28的图片，CNN 的输入则是一个3 x28 x28 的三维神经元。

- 卷积层
  
- 感受视野
  假设输入的是一个28x28的二维神经元，下面定义一个5x5的感受视野。隐藏层的神经元与输入层的5x5个神经元相连，这个5x5的区域就称为感受视野,这个结构可类似看作隐藏层中的神经元具有一个固定大小的感受视野去感受上一层的部分特征。
  
- 共享权值

  在同一个特征图上的神经元使用的卷积核是相同的，因此这些神经元共享权值，即共享卷积核中的权值和附带的偏移。若使用3个不同的卷积核，则可以输出3个特征图。
  
  [CNN](https://cs231n.github.io/convolutional-networks/)

- 激励层

  非线性映射

- 池化层 
  
  对特征图进行稀疏处理，最大池化与平均池化

  池化视野为$2$ x $ 2$，池化后深度不变，长宽变为原来一半
  
<img src="https://files.mdnice.com/user/38974/f814f19e-a0df-48a6-bc00-248b95c8e13a.png" alt="最大池化" >

<img src="https://files.mdnice.com/user/38974/e6074538-842b-4722-a2af-b39c9dca0277.png" height="200" width="600" >

- 全连接层

  把图像集转化成向量
  
  上图的$1$ x $100$向量为$12$ x $ 12$的卷积核卷积出来的

  全局平均值：最后一层的特征图直接求平均值。

- 输出层

  输出一个向量

<img src="https://files.mdnice.com/user/38974/3c02ff4e-117a-4a4c-b11e-4d13541609c9.png">

- 典型的神经网络  LeNet   AlexNet

Q:为什么卷积神经网络要用卷积，哪里卷了？

## pytorch简单实现神经网络应用

### Hello_nn_world

### 图片分类与识别

### [pytorch安装](https://www.bilibili.com/video/BV1cD4y1H7Tk/?spm_id_from=333.999.0.0)

Q:神经网络为什么被称为学习问题，而不是优化问题？我们发现神经网络无非拟合过程，为什么要单独赋予一个称呼？
  异或与非的问题
  非线性函数不采用多项式函数的问题

