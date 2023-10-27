
# -*- coding: utf-8 -*-
"""NN_pythorch.ipynb
by JDMV
"""

#import packages
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('./Coupled_NNs')
sys.path.append('./Data_for_figures')

from CDINN import *
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import time

# fixed the random seed
torch.manual_seed(73)
np.random.seed(73)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)



# Number of incident wave directions. (Maximum 64)
npts=64

directions=np.linspace(0,2.0*np.pi*(npts-1.0)/npts,npts,endpoint=True)
ds=directions.shape
directions=directions.reshape(ds[0],1)

t=np.linspace(0,2.0*np.pi*(64-1.0)/64,64,endpoint=True)

params=np.array([t for  i in range(len(directions))])


#Load the far-field pattern with 64 measurements, split into real and imaginary parts.
# with this far-field patter we will to reconstruc the shape in the online step
cases=[1,2,3,5]
for i in cases:
  testcase=np.load(f'./Data_for_figures/u_inf_Case{i}.npy')



  #Load the real boundary
  boundarytest=np.load(f'./Data_for_figures/boundarytestcase{i}.npy')


  # add noise to the far-field pattern data.
  sigma_ruido=0.01*max(abs(testcase[0])) # 1% of noise

  testcase_noise=testcase +np.random.normal(0,sigma_ruido,(64,128))

  #We select the far field pattern for each direction of incidence

  testcase_noise=testcase_noise[::int(64/npts)]

  # Set our inverse NN
  # The last layer contains the number of parameters to be recovered for the display form.

  layers=[64,64,128,64,10] #11 [a_0,...,a_5,b_1,...,b5,h,k]
  activations=[torch.nn.Tanh,torch.nn.Tanh,torch.nn.Tanh,torch.nn.Identity]

  inverse_NN=INN(layers,activations)

  #Online step. We can choose the form of representation (cosine or Fourier series)
  #%%time

  inverse_NN.train(params,testcase_noise,directions,representation='Cosine_series')

  coeficients_predicted,boundary_predicted=inverse_NN.predict(params)

  plt.plot(boundary_predicted[:,0],boundary_predicted[:,1],'^-',color='dimgray',
          label='Shape predicted')


  plt.plot(boundarytest[1::2],boundarytest[2::2],'o-',color='black',
          label='real shape',markersize=2.5)

  plt.axis('equal')
  plt.legend()
  plt.savefig(f'Approx_cosine_series_case{i}.eps')
  plt.show()
