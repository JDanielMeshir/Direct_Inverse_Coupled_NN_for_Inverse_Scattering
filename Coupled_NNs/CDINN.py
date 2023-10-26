#import packages
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import time

# fixed the random seed
np.random.seed(73)
torch.manual_seed(73)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# the direct deep neural network for direct scattering problem
class DNN(torch.nn.Module):
  def __init__(self):
    super(DNN, self).__init__()




    self.layers = [129,360,360,360,360,360,360,128]

    self.activation = [torch.nn.Tanh,torch.nn.Tanh,torch.nn.Tanh,torch.nn.Tanh,
                       torch.nn.Tanh,torch.nn.Tanh,torch.nn.Identity]

    self.depth = len(self.layers) - 1

    layer_list = list()

    for i in range(self.depth - 1):
        layer_list.append(
            ('layer_%d' % i, torch.nn.Linear(self.layers[i], self.layers[i+1]))
        )
        layer_list.append(('activation_%d' % i, self.activation[i]()))

    layer_list.append(
        ('layer_%d' % (self.depth - 1), torch.nn.Linear(self.layers[-2], self.layers[-1]))
    )
    layer_list.append(('activation_%d' % (self.depth - 1), self.activation[-1]()))

    # set up layer order dict

    layerDict = OrderedDict(layer_list)

    # deploy layers
    self.dnn = torch.nn.Sequential(layerDict).to(device)

    self.train_losses=[]
    self.test_losses=[]

  # ouput of DNN 
  def u_inf_net(self,X):
      u_inf=self.dnn(X)
      return u_inf

  def predict(self, X):
    X = torch.tensor(X).float().to(device)

    self.dnn.eval()
    u_inf=self.u_inf_net(X)
    u_inf=u_inf.detach().cpu().numpy()
    return u_inf


  def loss_func(self):

    self.optimizer.zero_grad()

    u_pred = self.u_inf_net(self.X_train_tensor)

    

    loss = torch.mean((self.Y_train_tensor - u_pred) ** 2)

    
    
    loss.backward()
    self.iter += 1
    u_val_pred=self.predict(self.X_test)

    loss_val=np.mean((self.Y_test-u_val_pred)**2)

    self.train_losses.append(loss.detach().cpu().numpy())
    self.test_losses.append(loss_val)

    if self.iter % 100 == 0:
      print('Iter %d, Loss: %.5e, Loss_val: %.5e'%(self.iter,loss,loss_val))
    return loss


  def train(self,X_train,Y_train,X_test,Y_test,Epochs=1000):

    self.X_test=X_test
    self.Y_test=Y_test
    # Backward and optimize


    self.iter=0
    self.X_train_tensor=torch.tensor(X_train).float().to(device)
    
    self.Y_train_tensor=torch.tensor(Y_train).float().to(device)

    self.dnn.train()
    self.optimizer = torch.optim.LBFGS(
          self.dnn.parameters(),
          lr=1.0,
          max_iter=60000,
          max_eval=60000,
          history_size=50,
          tolerance_grad=1e-15,
          tolerance_change=1.0 * np.finfo(float).eps,
          line_search_fn="strong_wolfe"       # can be "strong_wolfe"
      )
    for epoch in range(Epochs):
      self.dnn.train()
      self.optimizer.step(self.loss_func)
      
      
# the deep neural network for inverse problem
class INN(torch.nn.Module):
  def __init__(self, layers,activations):
    super(INN, self).__init__()

    self.activation = activations
    self.layers = layers

    npts=64
    self.depth = len(layers) - 1
    t = np.linspace(0,2.0*np.pi*(npts-1.0)/npts,npts,endpoint=True)
    self.t=torch.tensor(t).float().to(device)
    self.numberofparameters=int((self.layers[-1]-3)/2)

    be=[[torch.cos(ti),torch.sin(ti)] for ti in self.t]
    self.be=torch.tensor(be).to(device)
    # set up layer order dict

    layer_list = list()
    for i in range(self.depth - 1):
        layer_list.append(
            ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
        )
        layer_list.append(('activation_%d' % i, self.activation[i]()))

    layer_list.append(
        ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
    )
    layer_list.append(('activation_%d' % (self.depth - 1), self.activation[-1]()))


    layerDict = OrderedDict(layer_list)

    # deploy layers
    # Set the inverse NN
    self.inv_NN = torch.nn.Sequential(layerDict).to(device)

    # Here we can store the losse values. 
    self.train_losses=[]
    self.test_losses=[]

    ## Direct NN
    self.direct_NN=DNN()
    # Load the weigths found in offline step for direct NN

    self.direct_NN.dnn.load_state_dict(torch.load('./Coupled_NNs/Weigths_to_DNN.pt',
                                                  map_location=torch.device(device)))

  # set the boundary representation 

  def cosine_series_boundary_representation_out(self,out):

    xt=out[-2]+out[0]*torch.cos(self.t)

    for i in range(1,len(out)-5):
      xt=xt+out[i]*torch.cos((i+1.0)*self.t)
    xt+=out[-5]

    yt=out[-1]+out[-4]*torch.sin(self.t)+out[-3]
    m=torch.stack((xt,yt),dim=1)
    return m

  def Fourier_series_boundary_representation_out(self,out): 
    xt=torch.zeros_like(self.t)
    yt=torch.zeros_like(self.t)
    for i,ti in enumerate(self.t):
      rti=out[0]+sum([out[j]*torch.cos(j*ti)+out[j+self.numberofparameters]*torch.sin(j*ti)
                      for j in range(1,self.numberofparameters)])
      xt[i]=rti*torch.cos(ti)
      yt[i]=rti*torch.sin(ti)
    xt=xt+out[-2]
    yt=yt+out[-1]
    m=torch.stack((xt,yt),dim=1)
    return m


  def convv(self,outs):
    bes=[]
    for out in outs:
      rbe=self.boundary_representation_out(out)

      bes.append(torch.flatten(rbe))
    bes=torch.stack(bes).float().to(device)
    return bes

  #set loss function.

  def loss_func_cosine(self):

    self.optimizer.zero_grad()
    nn_out=self.inv_NN(self.X_train_tensor)
    nn_boundary=self.convv(nn_out)

    boundary_generated=torch.cat((self.directions, nn_boundary), 1)

    # evaluate in the direct NN the boundary generated with the ouput of the inverse NN

    nn_u_inf=self.direct_NN.dnn(boundary_generated)

    ############  loss function for Cosine_series representation ############
    ######## nn_u_inf= DNN((xt,yt),d) (DNN is the direct_NN trained) ########

    loss=torch.mean((self.Y_train_tensor-nn_u_inf)**2)

    ##########################################################################

    loss.backward()
    self.iter += 1

    self.train_losses.append(loss.detach().cpu().numpy())
    #self.test_losses.append(loss_val)
    if self.iter % 10 == 0:

      print('Iter %d, Loss: %.5e'%(self.iter,loss))
    return loss

  def loss_func_fourier(self):

    self.optimizer.zero_grad()
    nn_out=self.inv_NN(self.X_train_tensor)
    nn_boundary=self.convv(nn_out)
    boundary_generated=torch.cat((self.directions, nn_boundary), 1)
    nn_u_inf=self.direct_NN.dnn(boundary_generated)

    ab=[i**2*(nn_out[0][i]**2+nn_out[0][(self.numberofparameters)+i]**2)
     for i in range(1,self.numberofparameters+1)]



    loss_ab=sum(ab)
    ############  loss function for Fourier_series representation ############
    ######## nn_u_inf= DNN((xt,yt),d) (DNN is the direct_NN trained) ########

    loss=(torch.mean((self.Y_train_tensor-nn_u_inf)**2)+
          self.lamb1*loss_ab+self.lamb2*torch.abs(nn_out[0][0]))
    
    #########################################################################

    loss.backward()
    self.iter += 1

    self.train_losses.append(loss.detach().cpu().numpy())
    #self.test_losses.append(loss_val)
    if self.iter % 10 == 0:

      print('Iter %d, Loss: %.5e'%(self.iter,loss))
    return loss

  #Online step. Training the inverse NN

  def train(self,X_train,Y_train,directions,representation='Cosine_series',
            lamb1=0,lamb2=0):

    self.directions=torch.tensor(directions).float().to(device)
    self.lamb1=lamb1
    self.lamb2=lamb2

    #Optimizer LBFGS

    if representation=='Cosine_series':
      self.boundary_representation_out=self.cosine_series_boundary_representation_out

      self.iter=0
      self.X_train_tensor=torch.tensor(X_train).float().to(device)
      self.Y_train_tensor=torch.tensor(Y_train).float().to(device)
      self.inv_NN.train()
      self.optimizer = torch.optim.LBFGS(
            self.inv_NN.parameters(),
            lr=1.0,
            max_iter=60000,
            max_eval=60000,
            history_size=50,
            tolerance_grad=1e-15,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
      self.optimizer.step(self.loss_func_cosine)

    if representation=='Fourier_series':
      self.boundary_representation_out=self.Fourier_series_boundary_representation_out

      self.iter=0
      self.X_train_tensor=torch.tensor(X_train).float().to(device)
      self.Y_train_tensor=torch.tensor(Y_train).float().to(device)
      self.inv_NN.train()
      self.optimizer = torch.optim.LBFGS(
            self.inv_NN.parameters(),
            lr=1.0,
            max_iter=60000,
            max_eval=60000,
            history_size=50,
            tolerance_grad=1e-15,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
      self.optimizer.step(self.loss_func_fourier)

    # Backward and optimize


  def predict(self, X):
    X = torch.tensor(X).float().to(device)

    self.inv_NN.eval()
    coefficients_predicted=self.inv_NN(X)
    boundary_predicted=self.boundary_representation_out(coefficients_predicted[0])
    boundary_predicted=boundary_predicted.detach().cpu().numpy()
    coefficients_predicted=coefficients_predicted.detach().cpu().numpy()
    return coefficients_predicted,boundary_predicted
