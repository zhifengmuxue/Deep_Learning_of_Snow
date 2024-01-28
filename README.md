# Deep Learning of Snow

## **Introduction**

This is a self-built deep learning framework designed to provide learners in the field of artificial intelligence with a project that solidifies their understanding of the fundamentals of deep learning through reading and comprehending code. The core component of the framework is a tensor processing class called Variable, which has rich computational capabilities, including basic arithmetic operations and a wide range of specific operations required in deep learning. With this tensor processing class, users can easily implement various complex mathematical calculations, laying the foundation for building neural networks.

The Variable class is designed based on the concept of a computational graph and implements an efficient algorithm for forward propagation and differentiation. This algorithm can significantly improve the computational speed of the differentiation process while maintaining computational accuracy. Therefore, even when dealing with large-scale neural networks, the framework ensures efficient computational performance.

This project has already implemented fast differentiation for higher-order functions, allowing users to build neural networks using DLSnow. DLSnow supports not only traditional neural network structures but also various advanced network structures. Readers can create convolutional neural networks (CNNs), recurrent neural networks (RNNs), and many other networks through existing utility classes and frameworks.

In the process of defining a neural network, this project adopts the Define-by-Run approach. This approach allows users to dynamically define Variable and Function at runtime, greatly improving code readability and maintainability. Additionally, the project employs visualization techniques for computational graphs, using tools like graphviz to visually display the computation graph of functions. This helps users better understand the structure and computation process of neural networks and provides a basis for optimizing network performance.

## **Dependencies:**

- python					 3.10.x
- numpy                     1.24.x
- matplotlib                3.7.1
- graphviz                   9.0.0

## Folder Description

- **DLSnow:** Core framework and related tools.
- **net:** Implementation of a simple network. Readers can use the framework to create more networks.
- **spiral_dataset:** Spiral dataset, a simple training example.
- **tests:** Test files for the creation process.
- **VICG:** Visualization of the computation graph. Neural networks can be visualized using examples in the folder.



------

## **介绍：**

这是一个自主搭建的深度学习框架，旨在为人工智能领域的学习者提供一个通过阅读和理解代码，牢固对深度学习基础认知的项目。框架的核心部分是一个名为Variable的张量处理类，它具备丰富的运算功能，包括四则运算以及深度学习中所需求的大量特定运算。通过这个张量处理类，用户可以轻松地实现各种复杂的数学计算，为神经网络的构建奠定基础。

​		Variable类基于计算图（Computational  Graph）的设计理念，实现了高效的前向传播（Forward  Propagation）求导算法。这种算法能够在保证计算精度的同时，大幅提高求导过程的运算速度。因此，即使在处理大规模的神经网络时，该框架也能确保高效的计算性能。

​		本项目已经实现了高阶函数的快速求导，用户可以使用DLSnow进行神经网络的搭建。DLSnow不仅支持传统的神经网络结构，还支持各种高级的网络结构，读者可以通过已有的工具类和框架创建卷积神经网络（CNN）和循环神经网络（RNN），以及更多的其他网络。

​		在定义神经网络的过程中，本项目采用了Define-by-Run的方法。这种方法允许用户在运行时动态地定义Variable和Function，极大地提高了代码的可读性和可维护性。同时，项目还采用了可视化计算图的技术，通过graphviz等工具将函数的计算图直观地展示出来。这有助于用户更好地理解神经网络的结构和运算过程，为优化网络性能提供依据。

## **依赖库：**

- python					 3.10.x
- numpy                     1.24.x
- matplotlib                3.7.1
- graphviz                   9.0.0

## 文件夹介绍

- **DLSnow:** 核心框架，以及相关工具
- **net:** 实现的简单网络，读者可以使用框架完成更多网络创建
- **spiral_dataset:** 螺旋数据集，简单训练实例
- **tests:** 创建过程的测试文件
- **VICG:** 可视化计算图，可以通过文件夹中的例子将神经网络可视化