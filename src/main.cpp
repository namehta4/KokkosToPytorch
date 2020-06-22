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

//Training batch size
const int64_t N = 64;
//Tensor input dimension
const int64_t D_in = 1000;
//Tensor hidden dimension
const int64_t H = 100;
//NN output dimension
const int64_t D_out = 10;
//Total number of steps
const int64_t tstep = 1000;

using namespace torch;
using namespace std;
namespace py = pybind11;

typedef Kokkos::View<double *, Kokkos::LayoutRight> View1D;
typedef Kokkos::View<double **, Kokkos::LayoutRight> View2D;
typedef View2D::HostMirror h_View2D;


int main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    py::scoped_interpreter guard{};
    py::module sys = py::module::import("sys");
    sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
    py::module py_simplenn = py::module::import("py_simplenn");
    
    
    View2D foo1("Foo1", N,D_in);
    Kokkos::parallel_for(N*D_in, KOKKOS_LAMBDA(const int iter)
    {
      int i = iter / D_in;
      int j = iter % D_in;
      foo1(i,j) = i*j;
    });


// Calling simple NN implemented in Python
    py::object ob1 = py_simplenn.attr("run_NN")(py::array_t<double, py::array::c_style | py::array::forcecast>(N*D_in,foo1.data(),py::str{}),N,D_in,H,D_out,tstep);
    py::gil_scoped_release no_gil;

// Calling simple NN implemented in C++  
    FirstNN(N, D_in, H, D_out, tstep, foo1); 
  
  }
  Kokkos::finalize();
  return 0;
}


