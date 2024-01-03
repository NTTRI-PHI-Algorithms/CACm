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
import torch

def simulate_CAC(N,w,H0,K,parameters):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    dtype = torch.float64
    #dtype = torch.float32
    
    T = int(parameters['T'])
    T = torch.tensor(T).to(device)
    beta1 = torch.tensor(parameters['beta1'],dtype=dtype).to(device)
    beta2 = torch.tensor(parameters['beta2'],dtype=dtype).to(device)
    alpha = torch.tensor(parameters['alpha'],dtype=dtype).to(device)
    gamma = torch.tensor(parameters['gamma'],dtype=dtype).to(device)
    xi = torch.tensor(parameters['xi'],dtype=dtype).to(device)
    Dt = torch.tensor(parameters['Dt'],dtype=dtype).to(device)

    ### RETRIEVE PROBLEM ###
    w = torch.tensor(w,dtype=dtype).to(device)
    H0 = torch.tensor(H0*2).to(device)

    ### INITIALIZE ###
    N = torch.tensor(int(N)).to(device)
    K = torch.tensor(int(K)).to(device)
    x = (2*torch.rand(N, K, dtype=dtype)-1.0).to(device)
    s = torch.sign(x).to(device)
    e = torch.ones(N,K,dtype=dtype).to(device)

    mu = torch.mm(w,s)
    torch.cuda.synchronize()
    vH = -torch.sum(x*mu,dim=0)

    Hopt = torch.tensor(K,dtype=dtype).to(device)

    t = torch.tensor(int(0)).to(device)

    ### FOR SAVING ###
    reachedGS = torch.zeros(K,dtype=torch.int).to(device)

    p0, Ho, vH = simulate_CAC_run(N,K,T,t,beta1,beta2,alpha,gamma,xi,Dt,w,H0,x,e,s,vH,Hopt,reachedGS,dtype)

    return [(Ho.to("cpu")).numpy().astype(float)/2], (vH.to("cpu")).numpy().astype(float)/2, [(p0.to("cpu")).numpy()]


#current:
def simulate_CAC_run(N,K,T,t,beta1,beta2,alpha,gamma,xi,Dt,w,H0,x,e,s,vH,Hopt,reachedGS,dtype):

    xp = x
    
    ### ITERATE ###
    while t < T:

        ### ANALYZE ###
        reachedGS = reachedGS + (vH==H0).float()
        reachedGS = (reachedGS>0).float()

        ### ANNEAL ###
        beta = beta1 + t/T*(beta2-beta1) 
        
        ### SAVE PREVIOUS ###
        xpp = xp
        xp = x
       
        ### COUPLE ###
        y = torch.tanh(xp)
        mu = torch.mm(w,y)
        torch.cuda.synchronize()

        ### UPDATE STATE ###
        x = xp + Dt*( -beta*xp + alpha*e*mu + gamma*(xp-xpp)  )
        e = e - (xp**2-1.0)*e*xi
        
        if 1:
            norm = torch.mean(e,dim=0)
            e = e/norm
        e = torch.abs(e)
        
        s = torch.sign(x)
        
        ### ENERGY CALCULATION ###
        mu0 = torch.mm(w,s)
        torch.cuda.synchronize()

        vH = -torch.sum(s*mu0,dim=0)
        Hopt = torch.min(Hopt,vH)
            
        mvH = torch.mean(vH,dim=0)
        minvH = torch.min(vH,dim=0)

        t = t+1

    ### OUTPUT ###
    p0 = torch.mean(reachedGS)
    Ho = torch.min(Hopt)
    
    return p0, Ho, vH

