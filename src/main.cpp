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

int py2k2py(pybind11::array_t<double> pyrij)
{
  int N = 4;
  int D_in = 5;

  pybind11::buffer_info buf1 = pyrij.request();
  double *rij;
  double* h_rij = new double[3*3]; 
  cudaMalloc(&rij, 3*3*sizeof(double)); 

  h_rij = static_cast<double *>(buf1.ptr);
  cudaMemcpy(rij, h_rij, 3*3*sizeof(double), cudaMemcpyHostToDevice);
  
  double *d_x, *d_y;
  double* x = new double[N*D_in];
  double* y = new double[N*D_in];

  cudaMalloc(&d_x, N*D_in*sizeof(double)); 

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < D_in; j++)
      x[i*D_in+j] = i*j;
  }
  printf("Before: Value of X at 1,1 is %f \n",x[1*D_in+1]);

  cudaMemcpy(d_x, x, N*D_in*sizeof(double), cudaMemcpyHostToDevice);

// Calling simple NN implemented in Python
//  pybind11::scoped_interpreter guard{};
  pybind11::module sys = pybind11::module::import("sys");
  sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
  pybind11::module py_simplenn = pybind11::module::import("py_simple");
  pybind11::object ob1 = py_simplenn.attr("add_NN")(pybind11::array_t\
		  <double, pybind11::array::c_style | pybind11::array::forcecast>\
		  (N*D_in,d_x,pybind11::str{}),N,D_in);
  
  cudaDeviceSynchronize();
  cudaMemcpy(x,d_x, N*D_in*sizeof(double), cudaMemcpyDeviceToHost);
  printf("After: Value of X at 1,1 is %f \n",x[1*D_in+1]);
  
  cudaFree(rij);
  free(h_rij);
  cudaFree(d_x);
  free(x);
  free(y);
return 0;
}

using namespace std;

PYBIND11_MODULE(main, m)
{
  m.doc() = "pybind11 py2k2py plugin";
  m.def("py2k2py", &py2k2py, "Python 2 kokkos 2 python");
}

