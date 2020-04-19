# DROO

*Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks*

Python code to reproduce our DROO algorithm for Wireless-powered Mobile-Edge Computing [1], which uses the time-varying wireless channel gains as the input and generates the binary offloading decisions. It includes:

- [memory.py](memory.py): the DNN structure for the WPMEC, inclduing training structure and test structure, implemented based on [Tensorflow 1.x](https://www.tensorflow.org/install/pip).
  - [memoryTF2.py](memoryTF2.py): Implemented based on [Tensorflow 2](https://www.tensorflow.org/install).
  - [memoryPyTorch.py](memoryPyTorch.py): Implemented based on [PyTorch](https://pytorch.org/get-started/locally/).
- [optimization.py](optimization.py): solve the resource allocation problem

- [data](./data): all data are stored in this subdirectory, includes:

  - **data_#.mat**: training and testing data sets, where # = {10, 20, 30} is the user number

- [main.py](main.py): run this file for DROO, including setting system parameters, implemented based on [Tensorflow 1.x](https://www.tensorflow.org/install/pip)
  - [mainTF2.py](mainTF2.py): Implemented based on [Tensorflow 2](https://www.tensorflow.org/install).
  - [mainPyTorch.py](mainPyTorch.py): Implemented based on [PyTorch](https://pytorch.org/get-started/locally/).

- [demo_alternate_weights.py](demo_alternate_weights.py): run this file to evaluate the performance of DROO when WDs' weights are alternated

- [demo_on_off.py](demo_on_off.py): run this file to evaluate the performance of DROO when some WDs are randomly turning on/off


## Cite this work

1. L. Huang, S. Bi, and Y. J. Zhang, “[Deep reinforcement learning for online computation offloading in wireless powered mobile-edge computing networks](https://ieeexplore.ieee.org/document/8771176),” IEEE Trans. Mobile Compt., DOI:10.1109/TMC.2019.2928811, Jul. 2019.

## About authors

- [Liang HUANG](https://scholar.google.com/citations?user=NifLoZ4AAAAJ), lianghuang AT zjut.edu.cn

- [Suzhi BI](https://scholar.google.com/citations?user=uibqC-0AAAAJ), bsz AT szu.edu.cn

- [Ying Jun (Angela) Zhang](https://scholar.google.com/citations?user=iOb3wocAAAAJ), yjzhang AT ie.cuhk.edu.hk

## Required packages

- Tensorflow

- numpy

- scipy

## How the code works

- For DROO algorithm, run the file, [main.py](main.py). If you code with Tenforflow 2 or PyTorch, run [mainTF2.py](mainTF2.py) or [mainPyTorch.py](mainPyTorch.py), respectively.

- For more DROO demos:
  - Laternating-weight WDs, run the file, [demo_alternate_weights.py](demo_alternate_weights.py)
  - ON-OFF WDs, run the file, [demo_on_off.py](demo_on_off.py)
  - Remember to respectively edit the *import MemoryDNN* code from
    ```
      from memory import MemoryDNN
    ```
    to
    ```
      from memoryTF2 import MemoryDNN
    ```
    or
    ```
      from memoryPyTorch import MemoryDNN
    ```
    if you are using Tensorflow 2 or PyTorch.
    
### The DROO algorithm is coded based on [Tensorflow 1.x](https://www.tensorflow.org/install/pip). If you are fresh to deep learning, please start with [Tensorflow 2](https://www.tensorflow.org/install) or [PyTorch](https://pytorch.org/get-started/locally/), whose codes are much cleaner and easier to follow.
