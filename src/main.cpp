#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/embed.h>
#include<Kokkos_Core.hpp>
#include<iostream>
#include<random>
#include<cmath>
#include<cstdio>
#include<stdio.h>

#if defined(KOKKOS_ENABLE_CUDA)
#include<cuda.h>
#include<cuda_runtime.h>
#endif

#if defined(KOKKOS_ENABLE_HIP)
#include<hip/hip_runtime.h>
#endif

using namespace std;

typedef Kokkos::View<double *, Kokkos::LayoutRight> View1D;
typedef Kokkos::View<double **, Kokkos::LayoutRight> View2D;
typedef View2D::HostMirror h_View2D;

int py2k2py()
{
  int N = 64;
  int D_in = 1000;

  Kokkos::initialize();
  {
    View2D foo1("Foo1", N,D_in);
    h_View2D h_foo1;
    h_foo1 = create_mirror_view(foo1);

    Kokkos::parallel_for(N*D_in, KOKKOS_LAMBDA(const int iter)
    {
      int i = iter / D_in;
      int j = iter % D_in;
      foo1(i,j) = i*j;
    });
    deep_copy(h_foo1,foo1);
    printf("Before: Value of foo1(60,60) is %f \n",h_foo1(60,60));
    Kokkos::fence();


// Calling simple NN implemented in Python
//    pybind11::scoped_interpreter guard{};
    pybind11::module sys = pybind11::module::import("sys");
    sys.attr("path").attr("insert")(1, CUSTOM_SYS_PATH);
    pybind11::module py_simple = pybind11::module::import("py_simple");
    pybind11::object ob1 = py_simple.attr("add_NN")(pybind11::array_t\
		    <double, pybind11::array::c_style | pybind11::array::forcecast>\
		    (N*D_in,foo1.data(),pybind11::str{}),N,D_in);
    pybind11::gil_scoped_release no_gil;
    
    deep_copy(h_foo1,foo1);
    printf("After: Value of foo1(60,60) is %f \n",h_foo1(60,60));
  }
  Kokkos::finalize();
  return 0;
}

PYBIND11_MODULE(main, m)
{
  m.doc() = "pybind11 py2k2py plugin";
  m.def("py2k2py", &py2k2py, "Python 2 kokkos 2 python");
}

