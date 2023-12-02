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