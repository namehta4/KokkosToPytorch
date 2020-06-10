#include<torch/torch.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/embed.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<iostream>
#include<random>
#include<cmath>
#include<cstdio>
#include"FirstNN.h"

using namespace torch;
using namespace std;
namespace py = pybind11;


void FirstNN(const int N, const int D_in, const int H, const int D_out, const int tstep, Kokkos::View<double**, Kokkos::LayoutRight> foo1)
{
  cout << "*********************************************" << endl;
  cout << "Start C++" << endl;
  torch::manual_seed(1);
  torch::DeviceType device_type;
  if (torch::cuda::is_available() == 1)
  {
    std::cout << "CUDA is available! Training on GPU" << std::endl;
    device_type = torch::kCUDA;
  }
  else
  {
    std::cout << "Training on CPU" << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device = device_type;
//  torch::Device device(torch::kCUDA);

  auto options = torch::TensorOptions().device(device);    
  torch::Tensor X = torch::from_blob(foo1.data(),{N,D_in},options).to(device);
//  torch::Tensor X = torch::randn({N, D_in}).to(device);
  torch::Tensor Y = torch::randn({N, D_out}).to(device);
  
  TwoLayerNet net(D_in,H,D_out);
  net->to(device);

  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4));
  torch::nn::MSELoss criterion((torch::nn::MSELossOptions(torch::kSum)));

  for(int64_t ts=0;ts<=tstep;++ts)
  {
    torch::Tensor Y_pred = net->forward(X);
    torch::Tensor loss = criterion(Y_pred, Y);
    if(ts % 100 == 0)
    {
      printf("\r[%4ld/%4d] | D_loss: %e \n", ts,tstep,loss.item<float>());
    }
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
  }
  cout << "Training complete!" << endl;
  cout << "End C++" << endl;
  cout << "*********************************************" << endl;;
}


