# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:45:26 2018

@author: Administrator
"""
import numpy as np
from scipy import optimize
from scipy.special import lambertw
import scipy.io as sio                     # import scipy.io for .mat file I/
import time


def plot_gain( gain_his):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl
    
    gain_array = np.asarray(gain_his)
    df = pd.DataFrame(gain_his)
    
    
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    rolling_intv = 20

    plt.plot(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Gain ratio')
    plt.xlabel('learning steps')
    plt.show()
    
def bisection(h, M, weights=[]):
    # the bisection algorithm proposed by Suzhi BI
    # average time to find the optimal: 0.012535839796066284 s

    # parameters and equations
    o=100
    p=3
    u=0.7
    eta1=((u*p)**(1.0/3))/o
    ki=10**-26   
    eta2=u*p/10**-10
    B=2*10**6
    Vu=1.1
    epsilon=B/(Vu*np.log(2))
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]
    M1=np.where(M==1)[0]
    
    hi=np.array([h[i] for i in M0])
    hj=np.array([h[i] for i in M1])
    

    if len(weights) == 0:
        # default weights [1, 1.5, 1, 1.5, 1, 1.5, ...]
        weights = [1.5 if i%2==1 else 1 for i in range(len(M))]
        
    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    
    def sum_rate(x):
        sum1=sum(wi*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3))
        sum2=0
        for i in range(len(M1)):
            sum2+=wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1])
        return sum1+sum2

    def phi(v, j):
        return 1/(-1-1/(lambertw(-1/(np.exp( 1 + v/wj[j]/epsilon))).real))

    def p1(v):
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):
        sum1 = sum(wi*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            sum2 += wj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    def tau(v, j):
        return eta2*hj[j]**2*p1(v)*phi(v,j)

    # bisection starts here
    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v))
    for j in range(len(M1)):
        x.append(tau(v, j))

    return sum_rate(x), x[0], x[1:]



def cd_method(h):
    N = len(h)
    M0 = np.random.randint(2,size = N)
    gain0,a,Tj= bisection(h,M0)
    g_list = []
    M_list = []
    while True:
        for j in range(0,N):
            M = np.copy(M0)
            M[j] = (M[j]+1)%2
            gain,a,Tj= bisection(h,M)
            g_list.append(gain)
            M_list.append(M)
        g_max = max(g_list)
        if g_max > gain0:
            gain0 = g_max
            M0 = M_list[g_list.index(g_max)]
        else:
            break
    return gain0, M0


if __name__ == "__main__":
                
    h=np.array([6.06020304235508*10**-6,1.10331933767028*10**-5,1.00213540309998*10**-7,1.21610610942759*10**-6,1.96138838395145*10**-6,1.71456339592966*10**-6,5.24563569673585*10**-6,5.89530717142197*10**-7,4.07769429231962*10**-6,2.88333185798682*10**-6])
    M=np.array([1,0,0,0,1,0,0,0,0,0])
#    h=np.array([1.00213540309998*10**-7,1.10331933767028*10**-5,6.06020304235508*10**-6,1.21610610942759*10**-6,1.96138838395145*10**-6,1.71456339592966*10**-6,5.24563569673585*10**-6,5.89530717142197*10**-7,4.07769429231962*10**-6,2.88333185798682*10**-6])
#    M=np.array([0,0,1,0,1,0,0,0,0,0])

    
#    h = np.array([4.6368924987170947*10**-7,	1.3479411763648968*10**-7,	7.174945246007612*10**-6,	2.5590719803595445*10**-7,	3.3189928740379023*10**-6,	1.2109071327755575*10**-5,	2.394278475886022*10**-6,	2.179121774067472*10**-6,	5.5213902658478367*10**-8,	2.168778154948169*10**-7,	2.053227965874453*10**-6,	7.002952297466865*10**-8,	7.594077851181444*10**-8,	7.904048961975136*10**-7,	8.867218892023474*10**-7,	5.886007653360979*10**-6,	2.3470565740563855*10**-6,	1.387049627074303*10**-7,	3.359475870531776*10**-7,	2.633733784949562*10**-7,	2.189895264149453*10**-6,	1.129177795302099*10**-5,	1.1760290137191366*10**-6,	1.6588656719735275*10**-7,	1.383637788476638*10**-6,	1.4485928387351664*10**-6,	1.4262265958416598*10**-6, 1.1779725004265418*10**-6, 7.738218993031842*10**-7,	4.763534225174186*10**-6])
#    M =np.array( [0,	0,	1,	0, 0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	1,])
    
#    time the average speed of bisection algorithm
#    repeat = 1
#    M =np.random.randint(2, size=(repeat,len(h)))
#    start_time=time.time()
#    for i in range(repeat):
#        gain,a,Tj= bisection(h,M[i,:])
#    total_time=time.time()-start_time
#    print('time_cost:%s'%(total_time/repeat))
    
    gain,a,Tj= bisection(h,M)
    print('y:%s'%gain)
    print('a:%s'%a)
    print('Tj:%s'%Tj)
    
    # test CD method. Given h, generate the max mode
    gain0, M0 = cd_method(h)
    print('max y:%s'%gain0)
    print(M0)
    
    # test all data
    K = [10, 20, 30]                     # number of users
    N = 1000                     # number of channel
    
    
    for k in K:
            # Load data
        channel = sio.loadmat('./data/data_%d' %int(k))['input_h']
        gain = sio.loadmat('./data/data_%d' %int(k))['output_obj']
    
        start_time=time.time()
        gain_his = []
        gain_his_ratio = []
        mode_his = []
        for i in range(N):
            if i % (N//10) == 0:
               print("%0.1f"%(i/N))
               
            i_idx = i 
                
            h = channel[i_idx,:]
            
            # the CD method
            gain0, M0 = cd_method(h)
            
    
            # memorize the largest reward
            gain_his.append(gain0)
            gain_his_ratio.append(gain_his[-1] / gain[i_idx][0])
    
            mode_his.append(M0)
                
        
        total_time=time.time()-start_time
        print('time_cost:%s'%total_time)
        print('average time per channel:%s'%(total_time/N))
        
    
        plot_gain(gain_his_ratio)
        
            
        print("gain/max ratio: ", sum(gain_his_ratio)/N)

    































