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

//The batch size for training
const int64_t N = 64;
//The input dimension
const int64_t D_in = 1000;
//The hidden dimension
const int64_t H = 100;
//The output dimension
const int64_t D_out = 10;
//Total number of steps
const int64_t tstep = 1000;

using namespace torch;
using namespace std;
namespace py = pybind11;

typedef Kokkos::View<double *, Kokkos::LayoutRight> View1D;
typedef Kokkos::View<double **, Kokkos::LayoutRight> View2D;
typedef View2D::HostMirror h_View2D;

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



int main(int argc, char* argv[])
{
  torch::manual_seed(1);
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA is available! Training on GPU" << std::endl;
  }

  Kokkos::initialize(argc, argv);
  {
    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
    py::module py_simplenn = py::module::import("py_simplenn");
    
    
    View2D foo1("Foo1", N,D_in);
    h_View2D h_foo1 = Kokkos::create_mirror_view(foo1);
//    Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace> rand_pool(1313);
//    double number1 = rand_pool.normal();
//    cout << number1 << endl;
//    double number2 = rand_pool.get_state().normal();
//    cout << number2 << endl;
//    Kokkos::fill_random(foo1,rand_pool.get_state().normal(),N*D_in);
    Kokkos::parallel_for(N*D_in, KOKKOS_LAMBDA(const int iter)
    {
      int i = iter / D_in;
      int j = iter % D_in;
      foo1(i,j) = i*j;
    });
    Kokkos::deep_copy(h_foo1, foo1);
//Calling simple NN implemented in Python
    
    py::object ob1 = py_simplenn.attr("run_NN")(py::array_t<double, py::array::c_style | py::array::forcecast>(N*D_in,h_foo1.data()),N,D_in,H,D_out,tstep);
    py::gil_scoped_release no_gil;

//Simple NN implemented in C++   
    auto options = torch::TensorOptions().device(torch::kCUDA,0);    
    torch::Tensor X = torch::from_blob(foo1.data(),{N,D_in},options).to(device);
//    torch::Tensor X = torch::randn({N, D_in});
    torch::Tensor Y = torch::randn({N, D_out});
    
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
        printf("\r[%4ld/%4ld] | D_loss: %e \n", ts,tstep,loss.item<float>());
      }
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
    }


    std::cout<<"Training complete!"<<std::endl;
  }
  Kokkos::finalize();
  return 0;
}


