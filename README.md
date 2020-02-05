# DROO

*Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks*

Python code to reproduce our DROO algorithm for Wireless-powered Mobile-Edge Computing [1], which uses the time-varying wireless channel gains as the input and generates the binary offloading decisions. It includes:

- [memory.py](memory.py): the DNN structure for the WPMEC, inclduing training structure and test structure
- [optimization.py](optimization.py): solve the resource allocation problem

- [data](./data): all data are stored in this subdirectory, includes:

  - **data_#.mat**: training and testing data sets, where # = {10, 20, 30} is the user number

- [main.py](main.py): run this file for DROO, including setting system parameters

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

- For DROO algorithm, run the file, [main.py](main.py)

- For DROO demo with laternating-weight WDs, run the file, [demo_alternate_weights.py](demo_alternate_weights.py)

- For DROO demo with ON-OFF WDs, run the file, [demo_on_off.py](demo_on_off.py)

## The DROO algorithm is coded based on Tensorflow 1.\*. If you are are fresh to Tensorflow or already using Tensorflow 2, please refer to [DROO-tensorflow2](https://github.com/revenol/DROO-tensorflow2), whose codes are much cleaner and easier to follow. If you code with PyTorch, please refer to [DROO-PyTorch](https://github.com/revenol/DROO-PyTorch).
