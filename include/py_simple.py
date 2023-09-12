import torch
import sys
import numpy as np
import torch
import sys

class ReverseHolder():
    def __init__(self):
        self.tensor = tensor
        self.__array_interface__ = tensor.__cuda_array_interface__

class InterfaceHolder():
    def __init__(self,cuda_array_interface):
        self.__cuda_array_interface__ = cuda_array_interface

def add_NN(foo1, N, D_in):
    dtype = torch.float;
    if (torch.cuda.is_available() == 1):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
   
    #print("Starting py_simple.py",flush=True)
    #print("*********************************************",flush=True)
    interface=foo1.__array_interface__
    t = InterfaceHolder(interface)
    x = torch.as_tensor(t, device=device)
    x = x.view(D_in,N).transpose(0,1)
    
    y = torch.ones(N,D_in,device=device)
    x[:] = x+y
    torch.cuda.synchronize

    #print("Ending py_simple.py",flush=True)
    #print("*********************************************",flush=True)
    return None

