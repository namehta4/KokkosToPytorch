import torch
import numpy as np
import sys
sys.path.append('../include')
from test_kokkos import kokkos_loop
from test_kokkos import kokkos_begin
from test_kokkos import kokkos_end

class ReverseHolder():
    def __init__(self,tensor):
        self.tensor = tensor
        self.__array_interface__ = tensor.__cuda_array_interface__

class InterfaceHolder():
    def __init__(self,cuda_array_interface):
        self.__cuda_array_interface__ = cuda_array_interface


torch.set_default_dtype(torch.float64)
dtype = torch.float64
if (torch.cuda.is_available() == 1):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

kokkos_begin()

N = 10
D = 3

force_nn = torch.ones((N,D), device=device, dtype=torch.double)
t_fnn = ReverseHolder(force_nn)
print("From launch.py after initialize", force_nn[9][2], flush=True)

kokkos_loop(t_fnn, N, D)
print("From launch.py after C++ part", force_nn[9][2], flush=True)

kokkos_end()
