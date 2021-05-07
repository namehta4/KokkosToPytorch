import numpy as np
import torch
import sys

print(sys.version)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H)
        self.linear2 = torch.nn.Linear(H,D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

def run_NN(foo1, N, D_in, H, D_out, tstep):
    print("*********************************************")
    print("Start python")
    dtype = torch.float;
    if (torch.cuda.is_available() == 1):
        device = torch.device("cuda:0")
        print("CUDA is available! Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
   

    foo1 = np.reshape(foo1, (-1,D_in))

    x = torch.from_numpy(foo1).to(device)
    x = x.float()
#    x = torch.randn(N, D_in, dtype=dtype, device=device)
    y = torch.randn(N, D_out, dtype=dtype, device=device)
    
    model = TwoLayerNet(D_in, H, D_out)
    model = model.to(device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for t in range(tstep+1):
        y_pred = model(x/1e6)
        loss = criterion(y_pred, y)
        if t%100 == 0:
            print("[%4d/%4d] | D_loss: %12.6e"% (t,tstep,loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training complete!")
    print("End python")
    print("*********************************************")
