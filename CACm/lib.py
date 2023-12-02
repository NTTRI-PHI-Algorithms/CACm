import numpy as np
import pandas as pd

def LoadInstance(name,N,i):
    path = "./../Data/%s/" % name
    file = "%s_%d" % (name,i+1)
    pathfile = path + file
    
    file = open ( pathfile , 'r')
    idx = np.array([[float(num) for num in line.split(' ')] for line in file ])

    w = np.zeros((N,N))
    for i in range(len(idx[:,0])):
        w[int(idx[i,0])-1,int(idx[i,1])-1] = idx[i,2]

    w = w + w.T
    
    return w

def LoadOptimal(name,N):
    path = "./../Data/%s/" % name
    file = "%s_REF" % (name)
    pathfile = path + file
    file = open ( pathfile , 'r')
    H0 = np.array([[float(num) for num in line.split(' ')] for line in file ])
    return H0
