import torch
import numpy as np
from main import py2k2py

rij = np.ones((3,3))
rij.astype(float)
rij = torch.as_tensor(rij)
#rij = rij.to('cuda')

py2k2py(rij)
