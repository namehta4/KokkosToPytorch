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

using namespace torch;
using namespace std;
namespace py = pybind11;

struct TwoLayerNetImpl : nn::Module{
  TwoLayerNetImpl(int D_in, int H, int D_out)
      : linear1(nn::LinearOptions(D_in, H).bias(false)),
	linear2(nn::LinearOptions(H, D_out).bias(false))
  {
    register_module("linear1", linear1);
    register_module("linear2", linear2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::clamp_min(linear1(x),0);
    x = linear2(x);
    return x;
  }
  nn::Linear linear1, linear2;
};
TORCH_MODULE(TwoLayerNet);


void FirstNN(const int N, const int D_in, const int H, const int D_out, const int tstep, Kokkos::View<double**, Kokkos::LayoutRight> foo1);


