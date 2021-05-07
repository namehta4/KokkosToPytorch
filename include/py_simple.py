import numpy as np
import torch
import sys
import cupy as cp
import numba

print(sys.version)

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
   
    foo1 = cp.reshape(foo1, (-1,D_in))
   
    x = torch.as_tensor(foo1, device="cuda")
    print(x)
    y = torch.ones(N, D_in, dtype=dtype, device=device)
    y = y+x
    print(y)
    foo1 = cp.asarray(y).get()
    
    return foo1

    print("End python")
    print("*********************************************")
