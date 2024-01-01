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


                    