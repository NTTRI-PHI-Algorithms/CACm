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
import time
import models_pyTorch as models
import params
import lib

#########################################################################
# Define problem

#selection of median instance
Nlist = np.array([60,100,140,200])

#load instance and energy
namelist = ['WISHART_%d_0.80' % N for N in Nlist]

#########################################################################
# General parameters
    
#simulation parameters
K = 10**4
id0 = 0

#########################################################################
# Run
    
TTSCPUl = []
bestE = []
TTSl = []
p0l = []

for N, name in zip(Nlist,namelist):
    
    #load instance
    w    = lib.LoadInstance(name,N,id0)
    H0   = lib.LoadOptimal(name,N)[id0]
    eps0=np.mean(np.abs(w))

    #retrieve user defined parameters
    parameters_array = params.get_userdefparams(N,id0,['CAC'])
 
    #calculate random H
    x = np.random.uniform(0,1,size=[N,K])
    s = np.sign(2*x-1.0)
    h = np.matmul(w,s)
    mH = -0.5*np.sum(h*s,axis=1)

    for method, parameters in zip(['CAC'],parameters_array):

        #renorm weights for Wishart
        if name.split('_')[0] == 'WISHART':
            parameters['alpha'] = parameters['alpha']/eps0

        t0 = time.time()

        Hmin, Hf, p0 = models.simulate_CAC(N,w,H0,K,parameters)

        t1 = time.time()-t0

        if p0[0]>0:
            TTS = (np.log(1-0.99))/(np.log(1-p0[0]))*parameters['T']
            print('Ho/H0: %d/%d, P0: %0.04f, TTS: %d, T: %d, t: %0.2f' % (Hmin[0],H0[0],p0[0],TTS,parameters['T'],t1))
        else:
            TTS = np.nan
            print('Ho/H0: %d/%d, P0: %0.04f, TTS: NaN, T: %d, t: %0.2f' % (Hmin[0],H0[0],p0[0],parameters['T'],t1))

        Eb = (np.min(Hmin)-np.mean(mH))/(H0-np.mean(mH))

        TTSl.append(TTS)
        p0l.append(p0)
        TTSCPUl.append(t1)
        bestE.append(Eb)


                    