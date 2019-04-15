#  #################################################################
#  Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks
#
#  This file contains a demo evaluating the performance of DROO by randomly turning on/off some WDs. It loads the training samples from ./data/data_#.mat, where # denotes the number of active WDs in the MEC network. Note that, the maximum computation rate need be recomputed by solving (P2) once a WD is turned off/on.
#
#  References:
#  [1] 1. Liang Huang, Suzhi Bi, and Ying-jun Angela Zhang, “Deep Reinforcement Learning for Online Ofﬂoading in Wireless Powered Mobile-Edge Computing Networks”, submitted to IEEE Journal on Selected Areas in Communications.
#
# version 1.0 -- April 2019. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

from memory import MemoryDNN
from optimization import bisection
from main import plot_rate, save_to_txt

import time


def WD_off(channel, N_active, N):
    # turn off one WD
    if N_active > 5: # current we support half of WDs are off
        N_active = N_active - 1
        # set the (N-active-1)th channel to close to 0
        # since all channels in each time frame are randomly generated, we turn of the WD with greatest index
        channel[:,N_active] = channel[:, N_active] / 1000000 # a programming trick,such that we can recover its channel gain once the WD is turned on again.
        print("    The %dth WD is turned on."%(N_active +1))
            
    # update the expected maximum computation rate
    rate = sio.loadmat('./data/data_%d' %N_active)['output_obj']
    return channel, rate, N_active

def WD_on(channel, N_active, N):
    # turn on one WD
    if N_active < N:
        N_active = N_active + 1
        # recover (N_active-1)th channel 
        channel[:,N_active-1] = channel[:, N_active-1] * 1000000 
        print("    The %dth WD is turned on."%(N_active))
    
    # update the expected maximum computation  rate
    rate = sio.loadmat('./data/data_%d' %N_active)['output_obj']        
    return channel, rate, N_active


    

if __name__ == "__main__":
    ''' 
        This demo evaluate DROO for MEC networks where WDs can be occasionally turned off/on. After DROO converges, we randomly turn off on one WD at each time frame 6,000, 6,500, 7,000, and 7,500, and then turn them on at time frames 8,000, 8,500, and 9,000. At time frame 9,500 , we randomly turn off two WDs, resulting an MEC network with 8 acitve WDs.
    '''
    
    N = 10                     # number of users
    N_active = N               # number of effective users
    N_off = 0                  # number of off-users
    n = 10000                     # number of time frames, <= 10,000
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
    channel_bak = channel.copy()
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

    
    for i in range(n):
        # for dynamic number of WDs
        if i ==0.6*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_off(channel, N_active, N)
        if i ==0.65*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_off(channel, N_active, N)
        if i ==0.7*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_off(channel, N_active, N)
        if i ==0.75*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_off(channel, N_active, N)
        if i ==0.8*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_on(channel, N_active, N)
        if i ==0.85*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_on(channel, N_active, N)
        if i ==0.9*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_on(channel, N_active, N)
            channel, rate, N_active = WD_on(channel, N_active, N)
        if i == 0.95*n:
            print("At time frame %d:"%(i))
            channel, rate, N_active = WD_off(channel, N_active, N)
            channel, rate, N_active = WD_off(channel, N_active, N)
                
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
            r_list.append(bisection(h[0:N_active]/1000000, m[0:N_active])[0])

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


    
