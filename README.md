# DROO

*Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks*

Python code to reproduce our works on Wireless-powered Mobile-Edge Computing [1], which uses the wireless channel gains as the input and the binary computing mode selection results as the output of a deep neural network (DNN). It includes:

- [memory.py](memory.py): the DNN structure for the WPMEC, inclduing training structure and test structure

- [data](./data): all data are stored in this subdirectory, includes:

  - **data_#.mat**: training and testing data sets, where # = {10, 20, 30} is the user number

- [main.py](main.py): run this file, including setting system parameters


## About our works

1. Liang Huang, Suzhi Bi, and Ying-jun Angela Zhang, **Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks**, on [arxiv:1808.01977](https://arxiv.org/abs/1808.01977).

## About authors

- Liang HUANG, lianghuang AT zjut.edu.cn

- Suzhi BI, bsz AT szu.edu.cn

- Ying Jun (Angela) Zhang, yjzhang AT ie.cuhk.edu.hk

## Required packages

- Tensorflow

- numpy

- scipy

## How the code works

run the file, [main.py](main.py)
