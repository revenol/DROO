#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains a demo evaluating the performance of DROO with laternating-weight WDs. It loads the training samples with default WDs' weights from ./data/data_10.mat and with alternated weights from ./data/data_10_WeightsAlternated.mat. The channel gains in both files are the same. However, the optimal offloading mode, resource allocation, and the maximum computation rate in 'data_10_WeightsAlternated.mat' are recalculated since WDs' weights are alternated.
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-jun Angela Zhang, “Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks”, on arxiv:1808.01977
#
# version 1.0 -- April 2019. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

from memory import MemoryDNN
from optimization import bisection
from main import plot_rate, save_to_txt

import time


def alternate_weights(case_id=0):
    '''
    Alternate the weights of all WDs. Note that, the maximum computation rate need be recomputed by solving (P2) once any WD's weight is changed. 
    Input: case_id = 0 for default weights; case_id = 1 for alternated weights.
    Output: The alternated weights and the corresponding rate.
    '''
    # set alternated weights
    weights=[[1,1.5,1,1.5,1,1.5,1,1.5,1,1.5],[1.5,1,1.5,1,1.5,1,1.5,1,1.5,1]]
    
    # load the corresponding maximum computation rate 
    if case_id == 0:
        # by defaulst, case_id = 0
        rate = sio.loadmat('./data/data_10')['output_obj']
    else:
        # alternate weights for all WDs, case_id = 1
        rate = sio.loadmat('./data/data_10_WeightsAlternated')['output_obj']
    return weights[case_id], rate

if __name__ == "__main__":
    ''' 
        This demo evaluate DROO with laternating-weight WDs. We evaluate an extreme case by alternating the weights of all WDs between 1 and 1.5 at the same time, specifically, at time frame 6,000 and 8,000.
    '''
    
    N = 10                     # number of users
    n = 10000                # number of time frames, <= 10,000
    K = N                   # initialize K = N
    decoder_mode = 'OP'    # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024          # capacity of memory structure
    Delta = 32             # Update interval for adaptive K
    
    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    channel = sio.loadmat('./data/data_%d' %N)['input_h']
    rate = sio.loadmat('./data/data_%d' %N)['output_obj']
    
    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size

    split_idx = int(.8* len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8* n)) # training data size
    
    
    mem = MemoryDNN(net = [N, 120, 80, N],
                    learning_rate = 0.01,
                    training_interval=10, 
                    batch_size=128, 
                    memory_size=Memory
                    )

    start_time=time.time()
    
    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    h = channel[0,:]
    
    # initilize the weights by setting case_id = 0.
    weight, rate = alternate_weights(0)
    print("WD weights at time frame %d:"%(0), weight)
    
    
    for i in range(n):
        # for dynamic number of WDs
        if i ==0.6*n:
            weight, rate = alternate_weights(1)
            print("WD weights at time frame %d:"%(i), weight)
        if i ==0.8*n:
            weight, rate = alternate_weights(0)
            print("WD weights at time frame %d:"%(i), weight)

                
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1; 
            else:
                max_k = k_idx_his[-1] +1; 
            K = min(max_k +1, N)

        
        i_idx = i
        h = channel[i_idx,:]
        
        # the action selection must be either 'OP' or 'KNN'
        m_list = mem.decode(h, K, decoder_mode)
        
        r_list = []
        for m in m_list:
            # only acitve users are used to compute the rate
            r_list.append(bisection(h/1000000, m, weight)[0])

        # memorize the largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        # save the mode with largest reward
        mode_his.append(m_list[np.argmax(r_list)])
#        if i <0.6*n:
        # encode the mode with largest reward
        mem.encode(h, m_list[np.argmax(r_list)])
        

    total_time=time.time()-start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)
 
    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))
    
    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")


    
