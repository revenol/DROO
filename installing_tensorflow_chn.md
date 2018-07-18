# 如何安装 tensorflow

> Tensorflow 的官网提供了详细的[安装说明](https://www.tensorflow.org/install/)。本文介绍我们是如何在Windows操作系统上安装tensorflow。希望可以帮助tensorflow的新人快速上手。同时可以参考相关[博客](https://www.w3cschool.cn/tensorflow/tensorflow-lbqi2chw.html)。

1. 安装 Anaconda. 从anaconda的[官网](https://www.anaconda.com/download/)免费下载软件并安装。我们建议选择Python 3.6的版本，根据自己的Windows版本选择是[32位软件](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86.exe)还是[64位软件](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe)。关于anaconda的使用，包括相关软件 Anaconda Navigator， Jupyter notebook，和 Spyder等可以参考相关[博客](https://segmentfault.com/a/1190000011126204)。

2. 新建conda环境，命名为tensorflow，输入以下命令：

`C:> conda create -n tensorflow python=3.6`

3. 激活新建的tensorflow环境，输入以下命令：
```
C:> activate tensorflow

 (tensorflow)C:>  # 这里显示会更具系统不同而变化
```

4. 安装TensorFlow软件。可以选择CPU版本或者GPU版本。

CPU版本：

`(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow`

GPU显卡安装：

`(tensorflow)C:> pip install --ignore-installed --upgrade tensorflow-gpu `

5. 测验安装。新建命令行，依次输入以下命令

```
C:> activate tensorflow
$ Python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```
如果TensorFlow安装成功，会得到以下输出结果

`Hello, TensorFlow!`
