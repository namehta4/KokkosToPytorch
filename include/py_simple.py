import numpy as np
import torch
import sys
import cupy as cp
import numba

print(sys.version)

#def add_NN():
def add_NN(foo1, N, D_in):
    print("*********************************************")
    print("Start python")
    dtype = torch.float;
    if (torch.cuda.is_available() == 1):
        device = torch.device("cuda:0")
        print("CUDA is available! Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
   
    #foo1 = cp.reshape(foo1, (-1,D_in))
    #print(foo1.__cuda_array_interface__)
    print(foo1.__array_interface__)

    x = torch.as_tensor(foo1, device="cuda")
    x = x.view(4,5)
    print(x)
    print(x.__cuda_array_interface__)
    y = torch.ones(4, 5, dtype=dtype, device=device)
    x[:] = y+x
    print(x.__cuda_array_interface__)
    print(x)

    #Objective is to remove the get() from the next line
    #foo1 = cp.asarray(y).get()
    
    #foo1[:] = 
    #return foo1
    return None

    print("End python")
    print("*********************************************")
