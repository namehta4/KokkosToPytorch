#include<torch/torch.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/embed.h>
#include<iostream>
#include<random>
#include<cmath>
#include<cstdio>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace torch;
using namespace std;
namespace py = pybind11;

int main(int argc, char* argv[])
{
  int N = 4;
  int D_in = 5;

  double *d_x, *d_y;
  double* x = new double[N*D_in];

  cudaMalloc(&d_x, N*D_in*sizeof(double)); 

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D_in; j++)
      x[i*D_in+j] = i*j;
  }
  printf("Before: Value of X at 1,1 is %f \n",x[1*D_in+1]);

  cudaMemcpy(d_x, x, N*D_in*sizeof(double), cudaMemcpyHostToDevice);

// Calling simple NN implemented in Python
  pybind11::scoped_interpreter guard{};
  pybind11::module sys = pybind11::module::import("sys");
  sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
  pybind11::module py_simplenn = pybind11::module::import("py_simple");
  py::object ob1 = py_simplenn.attr("add_NN")(py::array_t<double, py::array::c_style | py::array::forcecast>(N*D_in,d_x,py::str{}),N,D_in);

  x = ob1.cast<py::array_t<double>>().mutable_data();
  cudaMemcpy(d_x, x, N*D_in*sizeof(double), cudaMemcpyHostToDevice);

// Lines commented out below represent the ideal case we want
//  d_x = ob1.cast<py::array_t<double>>().mutable_data();
//  cudaDeviceSynchronize();
//  cudaMemcpy(x,d_x, N*D_in*sizeof(double), cudaMemcpyDeviceToHost);
  
  printf("After: Value of X at 1,1 is %f \n",x[1*D_in+1]);
  
  cudaFree(d_x);

return 0;
}


