#include<iostream>
#include<random>
#include<cmath>
#include<cstdio>
#include<stdio.h>
#include<hip/hip_runtime.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/embed.h>

//using namespace torch;
//using namespace std;
//namespace py = pybind11;

int main(int argc, char* argv[])
{
  int N = 4;
  int D_in = 5;

  double *d_x, *d_y;
  double* x = new double[N*D_in];
  double* y = new double[N*D_in];

  hipMalloc(&d_x, N*D_in*sizeof(double)); 

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D_in; j++)
      x[i*D_in+j] = i*j;
  }
  printf("Before: Value of X at 1,1 is %f \n",x[1*D_in+1]);

  hipMemcpyHtoD(d_x, x, N*D_in*sizeof(double));

// Calling simple NN implemented in Python
  pybind11::scoped_interpreter guard{};
  pybind11::module sys = pybind11::module::import("sys");
  sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
  pybind11::module py_simplenn = pybind11::module::import("py_simple");
  pybind11::object ob1 = py_simplenn.attr("add_NN")(pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>(N*D_in,d_x,pybind11::str{}),N,D_in);
  
  hipDeviceSynchronize();
  hipMemcpyDtoH(x,d_x, N*D_in*sizeof(double));
  printf("After: Value of X at 1,1 is %f \n",x[1*D_in+1]);
  
  hipFree(d_x);
  free(x);
  free(y);
  return 0;
}


