import numpy as np
import torch
import numba

class InterfaceHolder():
    def __init__(self,cuda_array_interface):
        self.__cuda_array_interface__ = cuda_array_interface

def add_NN(foo1, N, D_in):
    print("*********************************************")
    print("Start python")
    dtype = torch.float;
    if (torch.cuda.is_available() == 1):
        device = torch.device("cuda:0")
        print("HIP is available! Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
   
    interface=foo1.__array_interface__
    t = InterfaceHolder(interface)
    x = torch.as_tensor(t, device=device)
    x = x.view(4,5)
    
    y = torch.ones(N,D_in,device=device)
    x[:] = x+y

    return None

    print("End python")
    print("*********************************************")
