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

def get_userdefparams(N,id0,methods):

    parameters_array = []
  
    #########################################################################
    # CAC

    if any(s=='CAC' for s in methods):

        if N==60:

            parameters_CAC={'T': 60, 'beta1': 1.185, 'beta2': 1.185, 'alpha': 0.17, 'gamma': 1.27, 'xi': 0.07, 'Dt': 0.5}
            
        if N==100:

            parameters_CAC={'T': 100, 'beta1': 1.74, 'beta2': 1.74, 'alpha': 0.167, 'gamma': 1.29, 'xi': 0.07, 'Dt': 0.5}
            
            
        if N==140:

            parameters_CAC={'T': 150, 'beta1': 1.44, 'beta2': 1.44, 'alpha': 0.12, 'gamma': 1.3, 'xi': 0.013, 'Dt': 0.5}

        if N==200:
     
            parameters_CAC={'T': 290, 'beta1': 1.52, 'beta2': 1.52, 'alpha': 0.12, 'gamma': 1.22, 'xi': 0.005, 'Dt': 0.5}
    
   
        parameters_array.append(parameters_CAC)
   
        
    return parameters_array