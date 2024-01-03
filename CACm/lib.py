## CACm
# Chaotic Amplitude Control with momentum
# Author: Timothee Leleu

##BSD 3-Clause License

#Copyright (c) 2023, NTT Research Inc., PHI labs, algorithms

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
