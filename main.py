#  #################################################################
#  Deep Q-learning for Wireless-powered Mobile Edge Computing.
#
#  This file contains the main code to train and test the DNN. It loads the training samples saved in ./data/data_#.mat, splits the samples into three parts (training, validation, and testing data constitutes 60%, 20% and 20%), trains the DNN with training and validation samples, and finally tests the DNN with test data.
#
#  Input: ./data/data_#.mat
#    Data samples are generated according to the CD method presented in [2]. THere are 30,000 samples saved in each ./data/data_#.mat, where # is the user number. Each data sample includes
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       energy broadcasting parameter   |    output_a           |
#  -----------------------------------------------------------------
#  |     transmit time of wireless device  |    output_tau         |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#  Output:
#    - Training Time: the time cost to train 18,000 independent data samples
#    - Testing Time: the time cost to compute predicted 6,000 computing mode
#    - Test Accuracy: the accuracy of the predicted mode selection. Please note that the mode selection accuracy is different from computation rate accuracy, since two different computing modes may leads to similar weighted sum computation rates. From our experience, the accuracy of weighted sum computation rate (evaluated as DNN/CD) is higher than the accuracy of computing mode selection.
#    - ./data/weights_biases.mat: parameters of the trained DNN, which are used to re-produce this trained DNN in MATLAB.
#    - ./data/Prediction_#.mat
#    Besides the test data samples, it also includes the predicted mode selection. Given DNN-predicted mode selection, the corresponding optimal weighted sum computation rate can be computed by solving (P1) in [1], which achieves over 99.9% of the CD method [2].
#  -----------------------------------------------------------------
#  |       wireless channel gain           |    input_h            |
#  -----------------------------------------------------------------
#  |       computing mode selection        |    output_mode        |
#  -----------------------------------------------------------------
#  |       DNN-predicted mode selection    |    output_mode_pred   |
#  -----------------------------------------------------------------
#  |      weighted sum computation rate    |    output_obj         |
#  -----------------------------------------------------------------
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Shengli Zhang, and Ying-jun Angela Zhang, Deep Neural Network for Computation Rate Maximization in Wireless Powered Mobile-Edge Computing Systems, submitted to IEEE Wireless Communications Letters.
#  [2] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” submitted for publication, available on-line at arxiv.org/abs/1708.08810.
#
# version 1.0 -- January 2018. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################

import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

from memory import MemoryDNN
from optimization import bisection

import time


def plot_gain( gain_his, rolling_intv = 50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl
    
    gain_array = np.asarray(gain_his)
    df = pd.DataFrame(gain_his)
    
    
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
#    rolling_intv = 20

    plt.plot(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Gain ratio')
    plt.xlabel('learning steps')
    plt.show()

def save_to_txt(gain_his, file_path):
    with open(file_path, 'w') as f:
        for gain in gain_his:
            f.write("%s \n" % gain)

if __name__ == "__main__":
    ''' 
        This algorithm generates K modes from DNN, and chooses with largest 
        reward. The mode with largest reward is stored in the memory, which is 
        further used to train the DNN.
    '''
    
    K = 30                     # number of users
    N = 30000                     # number of channel
    KNM = K                   # number of nearest modes
    decoder_mode = 'knm'
    Memory = 1024
    
    print('#user = %d, #channel=%d, K=%d, decoder = %s'%(K,N,KNM,decoder_mode))
    # Load data
    channel = sio.loadmat('./data/data_%d' %K)['input_h']
    gain = sio.loadmat('./data/data_%d' %K)['output_obj']
    
#    # we use the last 10000 date for test
#    channel[0:10000,:] = channel[20000:30000,:]
#    gain[0:10000,:] = gain[20000:30000,:]

    # increase h for better training
    channel = channel * 1000000
    
    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if N > total data size

    split_idx = int(.8* len(channel))
    num_test = min(len(channel) - split_idx, N - int(.8* N)) # training data size



    mem = MemoryDNN(net = [K, 120, 80, K],
                    learning_rate = 0.01,
                    training_interval=10, 
                    batch_size=128, 
                    memory_size=Memory
                    )

    start_time=time.time()
    
    gain_his = []
    gain_his_ratio = []
    mode_his = []
    knm_idx_his = []
    for i in range(N):
        if i % (N//10) == 0:
           print("%0.1f"%(i/N))

        if i < N - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - N + num_test + split_idx
            
        h = channel[i_idx,:]
        
        # the action selection must be either 'knm' or 'knn'
        m_list = mem.decode(h, KNM, decoder_mode)
        
        r_list = []
        for m in m_list:
            r_list.append(bisection(h/1000000, m)[0])

        # memorize the largest reward
        gain_his.append(np.max(r_list))
        gain_his_ratio.append(gain_his[-1] / gain[i_idx][0])
        # record the index of largest reward
        knm_idx_his.append(np.argmax(r_list))
        # encode the mode with largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        mode_his.append(m_list[np.argmax(r_list)])
        
    plot_gain(knm_idx_his)
    mem.plot_cost()
    
    
    total_time=time.time()-start_time
    print('time_cost:%s'%total_time)
    print('average time per channel:%s'%(total_time/N))
    

    plot_gain(gain_his_ratio)
    
        
    print("gain/max ratio: ", sum(gain_his_ratio[-num_test: -1])/num_test)

    save_to_txt(knm_idx_his, "knm_idx_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(gain_his_ratio, "gain_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")


    
